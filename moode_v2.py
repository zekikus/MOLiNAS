import os
import copy
import math
import torch
import random
import pickle
import numpy as np
from utils.losses import *
from utils.metrics import *
from utils.distances import *
from scipy.stats import qmc
from torchinfo import summary
from models.model import Model
from cell_module.ops import OPS
from torch.utils.data import DataLoader
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from utils.cell_multi_class_dataset import CellDataset

"""
MOODE - Version 2
-----------------



"""

class MOODE():

    def __init__(self, 
                seed = 42,
                pop_size = 20,
                mutation_factor = 0.5,
                crossover_prob = 0.5,
                predictor = None,
                boundary_fix_type = 'random',
                mutation_strategy = 'rand1',
                crossover_strategy = 'bin'):
        
        self.seed = seed
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_factor = mutation_factor
        self.boundary_fix_type = boundary_fix_type
        self.mutation_strategy = mutation_strategy
        self.crossover_strategy = crossover_strategy

        self.solNo = 0
        self.D_CL = 0.5
        self.T = 100
        self.cr = 0.85
        self.totalTrainedModel = 0
        self.input_size = (1, 3, 128, 128)
        self.MAX_SOL = 500
        self.NUM_EDGES = 9
        self.NUM_VERTICES = 7
        self.DIMENSIONS = 28
        self.MAX_NUM_CELL = 5
        self.OP_SPOTS = self.NUM_VERTICES - 2
        self.CELLS = [i for i in range(2, self.MAX_NUM_CELL + 1)] # 2, 3, 4, 5
        self.FILTERS = [2**i for i in range(3, 6)] # 8, 16, 32
        self.OPS = [idx for idx, op in enumerate(list(OPS.keys())[:-1])]

        self.all_models = {}
        self.archive = [] # Potentially Pareto Optimal List
        self.predictor = predictor
        self.nbr_ops = len(OPS) - 1
        self.cutoff = sum(self.nbr_ops**i for i in range(4))
        self.sampler = qmc.Sobol(d=self.DIMENSIONS)

    
    def seed_torch(self, seed=42):
        """
        This function sets the random seed for various libraries in order to ensure reproducibility of
        results in a PyTorch project.
        
        :param seed: The seed parameter is an integer value that is used to initialize the random number
        generator. By setting a fixed seed value, we can ensure that the random numbers generated during
        the execution of the program are always the same. This is useful for reproducibility of results,
        defaults to 42 (optional)
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def writePickle(self, data, name, path=None):
        # Write History
        with open(f"results/{path}/model_{name}.pkl", "wb") as pkl:
            pickle.dump(data, pkl)
    def init_rnd_nbr_generators(self):
        """
        This function initializes four random number generators with the same seed.
        """
        # Random Number Generators
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.sample_pop_rnd = np.random.RandomState(self.seed)
        self.init_pop_rnd = np.random.RandomState(self.seed)
        self.jumping_rnd = np.random.RandomState(self.seed)

    def sampling_with_rnd(self, k = 2):
        """
        This function generates a list of random samples with a specified size and range.
        
        :param k: The parameter "k" represents the number of samples to be generated for each individual
        in the population. In other words, if the population size is "n", then the total number of
        samples generated will be "n*k", defaults to 2 (optional)
        :return: a list of k * pop_size samples, where each sample is a list of DIMENSIONS random
        numbers between 0 and 1, generated using the uniform distribution.
        """
        return self.init_pop_rnd.uniform(low=0.0, high=1.0, size=(self.pop_size * k, self.DIMENSIONS)).tolist()
    
    def sampling_with_quasi(self, k = 2):
        """
        This function performs quasi-random sampling by generating a list of random numbers and
        returning it as a list.
        
        :param k: The parameter "k" is the multiplier used to determine the number of samples to be
        drawn from the population. Specifically, the function will draw k times the population size
        number of samples. For example, if the population size is 100 and k is 2, then the function will
        draw 200, defaults to 2 (optional)
        :return: The function `sampling_with_quasi` is returning a list of `k` samples from a
        quasi-random sequence of length `pop_size`. The samples are generated using the `random` method
        of the `sampler` object. The `n` parameter of the `random` method specifies the number of
        samples to generate. The `tolist` method is used to convert the generated samples to a list
        """
        return self.sampler.random(n = self.pop_size * k).tolist()

    def sampling(self, k = 2, method = 'random'):
        if method == 'random':
            return self.sampling_with_rnd()
        elif method == 'sobol':
            return self.sampling_with_quasi()

    def init_P0_population(self, method = 'random'):
        """
        This function initializes a population of models using a specified method (default is random)
        and returns the data of the models.
        
        :param method: The method parameter is an optional argument that specifies the method to use for
        initializing the population. The default value is 'random', which means that the population will
        be initialized with randomly generated individuals.
        """

        data = []
        counter = 0
        
        candidates = self.sampling()

        while counter < self.pop_size:
            chromosome = np.array(candidates.pop(0)) 
            config = self.vector_to_config(chromosome)
            model = Model(chromosome, config, self.CELLS[config[-2]], self.FILTERS[config[-1]], compile=False)       

            h = model.get_hash()
            model.encoding = model.encode(predictor_encoding='trunc_path', cutoff=self.cutoff)
            isSameSolution, sol = self.check_solution(model)
            if isSameSolution == False:
                model.solNo = self.solNo
                self.solNo += 1
                dic = self.get_model_dict(model, h)
                data.append(dic)

                counter += 1
                del model
            
            if len(candidates) == 0:
                candidates = self.sampling()
        
        return data

    def get_param_value(self, value, step_size):
        ranges = np.arange(start=0, stop=1, step=1/step_size)
        return np.where((value < ranges) == False)[0][-1]

    def vector_to_config(self, vector):
        """
        This function converts a numpy array to discrete values for a given solution vector.
        
        :param vector: A numpy array that represents a neural network architecture in a continuous space
        """

        try:
            config = np.zeros(self.DIMENSIONS, dtype='uint8')
            
            max_edges = int(((self.NUM_VERTICES) * (self.NUM_VERTICES - 1)) / 2)
            # Edges
            for idx in range(max_edges):
                config[idx] = self.get_param_value(vector[idx], 2)

            # Vertices - Ops
            for idx in range(max_edges, max_edges + self.NUM_VERTICES - 2):
                config[idx] = self.get_param_value(vector[idx], len(OPS) - 1)

            # Number of Cells
            idx = max_edges + self.NUM_VERTICES - 2
            config[idx] = self.get_param_value(vector[idx], len(self.CELLS))
            
            # Number of Filters 
            config[idx + 1] = self.get_param_value(vector[idx + 1], len(self.FILTERS))
        except Exception as e:
            print("Error...", vector, e)
        
        return config
    
    def get_model_dict(self, model, hash):
            return  {"solNo": model.solNo, "chromosome": model.chromosome, 
                    "config": model.config, "nbr_cell": model.nbr_cell,
                    "nbr_filters": model.nbr_filters, "is_compiled": model.is_compiled,
                    "objectives": model.objectives, "org_matrix": model.org_matrix,
                    "matrix": model.matrix, "org_ops": model.org_ops,
                    "ops": model.ops, "isFeasible": model.isFeasible,
                    "encoding": model.encoding, "hash": hash, "fitness_type": model.fitness_type,
                    "dispersion": model.dispersion}

    def dict_to_model(self, dict):
        model = Model(dict["chromosome"], dict['config'],
                      dict["nbr_cell"], dict["nbr_filters"],
                      False, dict["org_matrix"],
                      dict["org_ops"])
        
        model.objectives = dict["objectives"]
        model.encoding = dict["encoding"]
        model.hash = dict["hash"]
        model.solNo = dict["solNo"]
        model.fitness_type = dict["fitness_type"]
        model.dispersion = dict["dispersion"]

        return model
    
    def sample_population(self, pop, size = None):
        '''Samples 'size' individuals'''

        selection = self.sample_pop_rnd.choice(np.arange(len(pop)), size, replace=False)
        return [i['chromosome'] for i in np.array(pop)[selection]]

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation(self, current=None, best=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(self.population, size=3)
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(self.population, size=5)
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(self.population, size=2)
            mutant = self.mutation_rand1(best, r1, r2)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(self.population, size=4)
            mutant = self.mutation_rand2(best, r1, r2, r3, r4)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(self.population, size=2)
            mutant = self.mutation_currenttobest1(current, best, r1, r2)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(self.population, size=3)
            mutant = self.mutation_currenttobest1(r1, best, r2, r3)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.crossover_rnd.rand(self.DIMENSIONS) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.crossover_rnd.randint(0, self.DIMENSIONS)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''
            Performs the exponential crossover of DE
        '''
        n = self.crossover_rnd.randint(0, self.DIMENSIONS)
        L = 0
        while ((self.crossover_rnd.rand() < self.crossover_prob) and L < self.DIMENSIONS):
            idx = (n+L) % self.DIMENSIONS
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''
            Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self.crossover_exp(target, mutant)
        return offspring

    def get_opposite_model(self, chromosome, a = 0, b = 1):
        """
        This function returns an opposite model based on the given chromosome, with optional parameters
        for a and b.
        
        :param chromosome: a numpy array representing the genetic code of a neural network model
        :param a: The value of the first parameter used in the calculation of the opposite chromosome.
        It has a default value of 0, defaults to 0 (optional)
        :param b: The parameter "b" is a default value of 1 that is used in the calculation of the
        opposite chromosome. It is an optional parameter that can be overridden by passing a different
        value to the function, defaults to 1 (optional)
        :return: an instance of the Model class with opposite chromosome values, based on the input
        chromosome, a, and b values.
        """

        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            opposite_chromosome = np.array([a[idx] + b[idx] - c for idx, c in enumerate(chromosome)])
        else:
            opposite_chromosome = np.array([a + b - c for c in chromosome])
        
        opposite_chromosome = self.boundary_check(opposite_chromosome)
        config = self.vector_to_config(opposite_chromosome)
        opposite_model = Model(opposite_chromosome, config, self.CELLS[config[-2]], self.FILTERS[config[-1]], compile=False)
        h = opposite_model.get_hash()
        opposite_model.hash = h

        return opposite_model
    
    def init_OP0_population(self, P0):
        """
        This function initializes a population of opposite models and returns their data.
        :return: a list containing information about opposite models generated using the
        `get_opposite_model` method. The size of the list is determined by the
        `pop_size` attribute of the object calling the method.
        """

        data = []
        counter = 0
        
        while counter < self.pop_size:
            opposite_model = self.get_opposite_model(P0[counter]['chromosome'])

            h = opposite_model.hash
            # if h != () and h not in self.dic: Bu şartı opposite model için sağlamak mümkün değil
            opposite_model.encoding = opposite_model.encode(predictor_encoding='trunc_path', cutoff=self.cutoff)
            
            
            opposite_model.solNo = self.solNo
            self.solNo += 1
            dict = self.get_model_dict(opposite_model, h)
            data.append(dict)

            counter += 1
            del opposite_model
        
        return data

    # Objective 2 -> Model Complexity
    def get_obj_2(self, model, is_candidate = False):
        
        if model.isFeasible:
            flops = None
            if is_candidate:
                # For efficient memory usage
                tmp_model = copy.deepcopy(model)
                tmp_model.compile_model()
                flops = summary(tmp_model, input_size=self.input_size, device="cpu", verbose=0)
                del tmp_model
                return np.log10(flops.total_mult_adds)

            flops = summary(model, input_size=self.input_size, device="cpu", verbose=0)
            
            return np.log10(flops.total_mult_adds)
        else:
            return 1e4

    # Objective 1 -> Segmentation Performance
    def get_obj_1(self, model):
        
        if (model.is_compiled is None) or (not model.is_compiled):
            model.compile_model()
        
        #model.fitness_type = "ACTUAL"
        
        fitness, _, log = model.evaluate(train_loader, val_loader, loss_fn, metric_fn, device)
        if fitness != 1:
            model.fitness_type = "ACTUAL"
            self.totalTrainedModel += 1
            self.writePickle(model, model.solNo, path)
            with open(f"results/{path}/model_{model.solNo}.txt", "w") as f:
                f.write(log)
        return fitness

    def is_weakly_dominate(self, a, b):
        """
        Check if vector 'a' is weakly dominate to vector 'b'
        for a multi-objective optimization problem.

        Args:
        a (tuple or list): first vector representing a solution in the objective space
        b (tuple or list): second vector representing a solution in the objective space

        Returns:
        bool: True if 'a' is weakly dominate to 'b', False otherwise
        """

        if type(a) is dict and type(b) is dict:
            a = a['objectives']
            b = b['objectives']

        dominates = True
        atLeastOneBetter = False

        
        for i in range(len(a)):
            if(a[i] > b[i]):
                dominates = False
                break
            elif a[i] < b[i]:
                atLeastOneBetter = True

        return dominates and atLeastOneBetter

    def train_predictor(self, data = None):

        if data is None:
            xtrain = np.array([d['encoding'] for d in self.data]) # path encoding
            ytrain = np.array([d['objectives'][0] for d in self.data]) # validation loss

        self.predictor.fit(xtrain, ytrain)
        predictions = self.predictor.predict(xtrain)
        mse = mean_squared_error(ytrain, predictions)
        print("Predictor MSE:", mse)

    def init_eval_pop(self):

        print("Start Initialization...")
        P0 = self.init_P0_population()
        OP0 = self.init_OP0_population(P0)

        self.best_arch = P0[0]

        for idx, model in enumerate(P0):
            m = self.dict_to_model(model)
            m.fitness_type = "ACTUAL"
            m.objectives = [self.get_obj_1(m), self.get_obj_2(m)]
            model = self.get_model_dict(m, m.get_hash())
            self.all_models[m.solNo] = model
            P0[idx] = model

            if model['objectives'][0] <= self.best_arch['objectives'][0]:
                self.best_arch = model

        for idx, model in enumerate(OP0):
            m = self.dict_to_model(model)
            m.fitness_type = "ACTUAL"
            m.objectives = [self.get_obj_1(m), self.get_obj_2(m)]
            model = self.get_model_dict(m, m.get_hash())
            self.all_models[m.solNo] = model
            OP0[idx] = model

            if model['objectives'][0] <= self.best_arch['objectives'][0]:
                self.best_arch = model

        self.data.extend(copy.deepcopy(P0))
        self.data.extend(copy.deepcopy(OP0))

        # return 
        return P0, OP0
        
    def gde3_selection(self, sol1, sol2):
        
        """
        Case 1: In the case of infeasible vectors, the trial vector is selected if it weakly
        dominates the old vector in constraint violation space, otherwise the
        old vector is selected.
        """
        if sol1['isFeasible'] == False and sol2['isFeasible'] == False:
            if self.is_weakly_dominate(sol2, sol1):
                return [sol2]
            else:
                return [sol1]
        
        elif sol1['isFeasible'] == False or sol2['isFeasible'] == False:
            """
            Case 2: In the case of the feasible and infeasible vectors, the feasible vector is
            selected.
            """
            if sol1['isFeasible']:
                return [sol1]
            else:
                return [sol2]
        
        elif sol1['isFeasible'] and sol2['isFeasible']:
            
            if self.is_weakly_dominate(sol2, sol1):
                """
                If both vectors are feasible, then the trial is selected if it weakly
                dominates the old vector in the objective function space.
                """
                return [sol2]
            elif self.is_weakly_dominate(sol1, sol2):
                """
                If the old vector dominates the trial vector, then the old vector is
                selected.
                """
                return [sol1]
            else:
                """
                If neither vector dominates each other in the objective function space,
                then both vectors are selected for the next generation.
                """
                return [sol1, sol2]

    def boundary_check(self, vector):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.
        projection == The invalid value is truncated to the nearest limit
        random == The invalid value is repaired by computing a random number between its established limits
        reflection == The invalid value by computing the scaled difference of the exceeded bound multiplied by two minus
        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector

        vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))

        return vector

    def create_new_model(self, sol):
        # Dictionary to Model
        config = self.vector_to_config(sol)
        sol = Model(sol, config, self.CELLS[config[-2]], self.FILTERS[config[-1]], compile=False)
        sol.encoding = sol.encode(predictor_encoding='trunc_path', cutoff=self.cutoff)
        h = sol.get_hash()
        sol.hash = h
        self.solNo += 1
        sol.solNo = self.solNo

        return sol

    def generate_candidate(self, X):
        X = X['chromosome']
        X = copy.deepcopy(X)
        V = self.mutation(current=X)
        U = self.crossover(X, V)
        U = self.boundary_check(U)
        U = self.create_new_model(U) # model
        return U

    def check_solution(self, model):

        model_dict = self.get_model_dict(model, model.get_hash())
        for i in self.all_models.keys():
            model_2 = self.all_models[i]
            D = jackard_distance_caz(model_dict, model_2)
            if D == 0:
                return True, model_2
        
        return False, None
    
    def evaluate_solution(self, U):
        isSameSolution, sol = self.check_solution(U)
        if isSameSolution:
            print("SAME SOLUTION")
            same_sol = self.all_models[sol['solNo']]
            U = self.dict_to_model(copy.deepcopy(same_sol))
        else:
            #U.solNo = self.solNo
            U.objectives = [self.get_obj_1(U), self.get_obj_2(U)]
            U.fitness_type = "ACTUAL"
            dict = self.get_model_dict(U, U.get_hash())
            self.all_models[U.solNo] = dict
            self.data.append(dict)
        
        
        return U, self.totalTrainedModel >= self.MAX_SOL

    def is_strictly_dominate(self, sol1, sol2):

        if type(sol1) is dict:
            a = sol1['objectives']
            b = sol2['objectives']
        else:
            a = sol1.objectives
            b = sol2.objectives

        if a[0] < b[0] and a[1] < b[1]:
            return True
        
        return False

    def update_archive(self, population):

        self.archive.extend(population)

        # Remove duplicated ones
        temp_idxs = []
        unique_sols = set()
        for idx, sol in enumerate(self.archive):
            if sol['solNo'] not in unique_sols:
                temp_idxs.append(idx)
            
            unique_sols.add(sol['solNo'])

        self.archive = np.array(self.archive)[temp_idxs].tolist()
        # Remove duplicated ones

        # Reset Archive solution domination count
        for a in self.archive:
            a["domination_count"] = 0

        for i, _ in enumerate(self.archive):
            for j, _ in enumerate(self.archive):
                if i == j: continue
                if self.is_weakly_dominate(self.archive[i], self.archive[j]):
                    self.archive[j]["domination_count"] += 1

        temp_archive = []
        for a in self.archive:
            if a["domination_count"] < 1:
                temp_archive.append(a)

        return temp_archive
    
    def calculate_predicted_value(self, candidate):
        predicted_obj_1 = self.predictor.predict([candidate.encoding])[0]
        predictions = np.array([tree.predict([candidate.encoding]) for tree in self.predictor]).T
        variance_pred = np.std(predictions)

        return predicted_obj_1, variance_pred


    def get_solution_reliability(self, D_CL, C, method = 'SA', D_t = 0.5):

        is_accepted = False
        if method == 'SA':
            delta_D = C.dispersion - D_CL
            p_acc = min(1, math.exp(-delta_D / self.T))

            if delta_D < 0:
                D_CL = C.dispersion
                is_accepted = True
            elif p_acc <= random.random():
                D_CL = C.dispersion
                is_accepted = True  
        else:
            delta_D = C.dispersion - D_CL
            if delta_D < 0:
                D_CL = C.dispersion
                D_t -= delta_D
                is_accepted = True
            else:
                if D_t - delta_D >= 0:
                    D_CL = C.dispersion
                    D_t -= delta_D
                    is_accepted = True


        return is_accepted

    def dominated_by_solution_in_archive(self, C):
        for sol in self.archive:
            if self.is_strictly_dominate(sol, C):
                return True
        
        return False

    def reliability_based_selection(self, D_CL, X, method = 'SA', D_t = 0.5, k = 10):
        
        counter = 0
        min_segmentation_error = sorted(self.population, key=lambda x: x['objectives'][0])[0]['objectives'][0]

        while counter < k:
            C = self.generate_candidate(X)
            while C.isFeasible == False:
                C = self.generate_candidate(X)

            # Calculate objectives for candidate solution
            predicted_obj_1, C.dispersion = self.calculate_predicted_value(C)
            C.fitness_type = "ESTIMATED"
            C.objectives = [predicted_obj_1, self.get_obj_2(C, is_candidate=True)]
            
            is_accepted = self.get_solution_reliability(D_CL, C, method=method)
            if is_accepted:

                # Rule 1:
                if self.dominated_by_solution_in_archive(self.get_model_dict(C, C.hash)) == False:
                    return [self.get_model_dict(C, C.hash)]
                # Rule 2
                elif C.objectives[0] < min_segmentation_error:
                    return [self.get_model_dict(C, C.hash)]
                # Rule 3
                elif self.is_weakly_dominate(self.get_model_dict(C, C.hash), X):
                    return [self.get_model_dict(C, C.hash)]

            counter += 1

        return [X]

    def evolve_generation(self):
        query = self.totalTrainedModel

        G = 1
        N = 5
        while self.totalTrainedModel <= self.MAX_SOL:
            if G < N:
                
                i = 0
                P_G = []
                while len(P_G) < self.pop_size:
                    X_i = self.population[i]['chromosome']
                    X_i = copy.deepcopy(X_i)
                    V_i = self.mutation(current=X_i)
                    U_i = self.crossover(X_i, V_i)
                    U_i = self.boundary_check(U_i)
                    U_i = self.create_new_model(U_i) # model

                    # Selection
                    X_i = self.dict_to_model(copy.deepcopy(self.population[i])) # target - model
                    # Evaluation
                    U_i, stopping_condition = self.evaluate_solution(U_i) # model

                    selected_sols = self.gde3_selection(self.get_model_dict(X_i, X_i.hash), self.get_model_dict(U_i, U_i.hash))
                    for sol in selected_sols:
                        P_G.append(copy.deepcopy(sol))
                    
                    # Stopping Condition
                    if stopping_condition:
                        return
                    
                    i += 1
                    del U_i
                
                self.population = P_G
                G += 1
            else:
                # Line 26 - 31
                i = 0
                P_G = []
                while len(P_G) < self.pop_size:
                    X_i = copy.deepcopy(self.population[i])
                    selected_sols = self.reliability_based_selection(self.D_CL, X_i, method="MSA")

                    for sol in selected_sols:
                        P_G.append(sol)
                    
                    del X_i
                    i += 1

                # Line 32 - 33
                for idx, sol in enumerate(P_G):
                    try:
                        if sol['fitness_type'] == "ESTIMATED":
                            sol, stopping_condition = self.evaluate_solution(self.dict_to_model(sol))
                            P_G[idx] = self.get_model_dict(sol, sol.hash)

                            # Stopping Condition
                            if stopping_condition:
                                return
                    except:
                        print()

                
                self.population = P_G
                self.train_predictor()

            # Update Archive
            self.archive = self.update_archive(self.population)
            self.T = self.T * self.cr

    def start_algorithm(self):
        self.init_rnd_nbr_generators()

        self.data = []

        # Line 1 - 8
        P0, OP0 = self.init_eval_pop()
        
        # Line 9
        self.train_predictor()

        # Line 10 - 14
        P_T = []
        counter = 0
        while len(P_T) < self.pop_size:
            selected_sol = self.gde3_selection(P0[counter], OP0[counter])
            P_T.extend(copy.deepcopy(selected_sol))
            counter += 1

        # Line 15
        self.population = P_T

        # Update Archive
        self.archive = self.update_archive(self.population)

        # Line 18 - 34
        self.evolve_generation()

        # Write Pareto Front
        for sol in self.archive:
            model = self.dict_to_model(sol)
            model.compile_model()
            self.writePickle(model, model.solNo, nds_path)


if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore")
    device = torch.device('cuda')
    
    predictor_method = 'boosted_decision_tree'
    data_path = "Datasets/WRTC/images_train_patches_pickle"
    batch_size = 128
    seed = 42

    train_dataset = CellDataset(data_path, mode="training", split=0.9, de_train=True)
    val_dataset = CellDataset(data_path, mode="training", split=0.9, is_val=True, de_train=True)

    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, pin_memory=True, drop_last=True)

    print("Number of training image:", train_dataset.__len__())

    path = f"moode_v2_{seed}"
    if not os.path.exists(f"results/{path}/"):
        os.makedirs(f"results/{path}/")

    nds_path = f"nds_v2_{seed}"
    if not os.path.exists(f"results/{nds_path}/"):
        os.makedirs(f"results/{nds_path}/")

    if predictor_method == 'boosted_decision_tree':
        predictor = AdaBoostRegressor(n_estimators=100)

    loss_fn = nn.CrossEntropyLoss()
    metric_fn = F1Score()

    predictors = {"boosted_decision_tree": AdaBoostRegressor(n_estimators=100)}

    moode = MOODE(predictor=predictors[predictor_method])
    moode.seed_torch(seed)
    moode.start_algorithm()
