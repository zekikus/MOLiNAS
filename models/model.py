import copy
import torch
import timeit
import torch.nn as nn
import numpy as np
from utils.early_stopping import EarlyStopping
import torch.optim as optim
from cell_module.encoder_cell import EncoderCell
from cell_module.decoder_cell import DecoderCell
from cell_module.ops import OPS as ops_dict
from utils.encodings import *

class Model(nn.Module):

    def __init__(self, chromosome = None, 
                       config = None, 
                       nbr_cell = None, 
                       nbr_filters = None, 
                       compile = True, 
                       matrix = None, 
                       ops = None):
                       
        super(Model, self).__init__()
        # CONSTANT
        self.NUM_VERTICES = 7
        self.MAX_EDGE_NBR = int(((self.NUM_VERTICES) * (self.NUM_VERTICES - 1)) / 2)

        self.dispersion = None
        self.fitness_type = None 
        self.is_compiled = None
        self.solNo = None
        self.objectives = [1, 1e4]
        self.cost = -1
        self.config = config
        self.nbr_cell = nbr_cell
        self.nbr_filters = nbr_filters
        self.chromosome = chromosome

        if (matrix is None) and (ops is None):
            self.org_matrix = self.create_matrix()
            self.org_ops = list(self.config[self.MAX_EDGE_NBR: self.MAX_EDGE_NBR + self.NUM_VERTICES - 2])
        else:
            self.org_matrix = matrix
            self.org_ops = ops

        self.cells = nn.ModuleList([])
        self.mp = nn.MaxPool2d((2, 2))
        
        self.matrix, self.ops, self.isFeasible = self.prune(self.org_matrix, self.org_ops)
        if self.isFeasible and compile:
            # Get ops except for Input and output nodes
            self.ops = self.ops[1:-1]
            self.compile()
            self.is_compiled = True
            
            for param in self.parameters():
                param.requires_grad = True
                if len(param.shape) > 1:
                    torch.nn.init.xavier_uniform_(param)
                
    
    def compile_model(self):
        if self.isFeasible and not self.is_compiled:
            # Get ops except for Input and output nodes
            self.ops = self.ops[1:-1]
            self.compile()
            self.is_compiled = True
            
            for param in self.parameters():
                param.requires_grad = True
                if len(param.shape) > 1:
                    torch.nn.init.xavier_uniform_(param)


    def create_matrix(self):
        """ Convert encoding vector to adjacency matrix """
        triu_indices = np.triu_indices(self.NUM_VERTICES, k = 1)
        self.matrix = np.zeros((self.NUM_VERTICES, self.NUM_VERTICES)).astype('int8')
        self.matrix[triu_indices] = self.config[0: self.MAX_EDGE_NBR]
        
        return self.matrix

    def compile(self):

        if len(self.cells) != 0:
            self.cells = nn.ModuleList([])

        """ Build U-Net Model """
        C_in = 3
        C_out = self.nbr_filters

        # Encoder Cell
        for cell_idx in range(self.nbr_cell):
            cell = EncoderCell(self.matrix, self.ops, C_in, C_out)
            C_in = C_out
            C_out = C_out * 2
            self.cells.append(cell)

        # Bottleneck
        self.cells.append(ops_dict['bottleneck'](C_in, C_out))

        # Decoder Cell
        for _ in range(self.nbr_cell):
            cell = DecoderCell(self.matrix, self.ops, C_in + C_in, C_in)
            self.cells.append(cell)
            C_in = C_in // 2
            C_out = C_out // 2
        
        # Output
        self.cells.append(nn.Sequential(nn.Conv2d(self.nbr_filters, 4, kernel_size=1, padding=0), nn.Identity(), nn.Softmax()))

    def forward(self, inputs):
        
        outputs = [0] * (self.nbr_cell * 2 + 2) # Cell outputs
        pool_outputs = [0] * len(outputs) # Pooling outputs
        outputs[0] = inputs 

        # Encoder Cells
        cell_nbr = 1
        _input = inputs
        while cell_nbr <= self.nbr_cell:
            cell_output = self.cells[cell_nbr - 1](_input) # Feed forward input to Cell
            outputs[cell_nbr] = cell_output # Store output of Cell
            pool_outputs[cell_nbr] = self.mp(cell_output) # Apply Max Pooling
            _input = pool_outputs[cell_nbr] # Store output of Pooling operation

            cell_nbr += 1

        # Bottleneck
        outputs[cell_nbr] = self.cells[cell_nbr - 1](pool_outputs[cell_nbr - 1])
        cell_nbr += 1
        
        # Decoder Cells
        skip_idx = cell_nbr - 2
        while cell_nbr < len(self.cells):
            _input = outputs[cell_nbr - 1]
            cell_output = self.cells[cell_nbr - 1](_input, outputs[skip_idx])
            outputs[cell_nbr] = cell_output 

            skip_idx -= 1
            cell_nbr += 1
        
        # Output
        outputs.append(self.cells[-1](outputs[-1]))

        return outputs[-1]
    
    def evaluate(self, train_loader, val_loader, loss_fn, metric_fn, device):
        
        try:
            print(f"Model {self.solNo} Training...")
            self.to(device) # cuda start

            train_loss = []
            train_f1score = []
            log = f"Model No: {self.solNo}\n"
            early_stopping = EarlyStopping(patience=10)

            startTime = timeit.default_timer()
            optimizer = optim.Adam(self.parameters(), lr=0.0001)
            for epoch in range(60):

                # Train Phase
                self.train()
                for inputs, labels in train_loader:
                    #inputs, labels = inputs.permute((0, 3, 2, 1)), labels.permute((0, 3, 2, 1))
                    inputs, labels = inputs.to(device), labels.to(device)
        
                    with torch.set_grad_enabled(True):
                        output = self.forward(inputs)
                        error = loss_fn(output, labels)
                        train_loss.append(error.item())
                        train_f1score.append(metric_fn(output, labels).item())
                        optimizer.zero_grad()
                        error.backward()
                        optimizer.step()
                
                torch.cuda.empty_cache()
		
                # Validation Phase
                val_loss = []
                val_dice = []
                self.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        #inputs, labels = inputs.permute((0, 3, 2, 1)), labels.permute((0, 3, 2, 1))
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = self.forward(inputs)
                        error = loss_fn(output, labels)
                        val_dice.append(metric_fn(output, labels).item())
                        val_loss.append(error)
                
                torch.cuda.empty_cache()
                
                # Log
                avg_tr_loss = sum(train_loss) / len(train_loss)
                avg_tr_score = sum(train_f1score) / len(train_f1score)
                avg_val_loss = sum(val_loss) / len(val_loss)
                avg_val_score = sum(val_dice) / len(val_dice)
                txt = f"\nEpoch: {epoch}, tr_loss: {avg_tr_loss}, tr_f1: {avg_tr_score}, val_loss: {avg_val_loss}, val_f1: {avg_val_score}"
                log += txt
                print(txt)

                # Early Stopping Check
                if early_stopping.stopTraining(epoch, avg_val_loss, avg_val_score):
                    self.objectives[0] = 1 - early_stopping.best_score
                    self.cost = timeit.default_timer() - startTime
                    print(f"Stop Training - Model {self.solNo} , {self.objectives[0]}, {self.cost}")
                    break
            
        except Exception as e: # Memory Problems
            torch.cuda.empty_cache()
            print(e)
            return 1, 1e4, None

        torch.cuda.empty_cache()

        self.objectives[0] = 1 - early_stopping.best_score
        self.cost = timeit.default_timer() - startTime
        
        log += f"\nElapsed Time: {self.cost}, Fitness: {self.objectives[0]}"
        
        return self.objectives[0], self.cost, log

    def prune(self, original_matrix, original_ops):

        """Prune the extraneous parts of the graph.
        General procedure:
        1) Remove parts of graph not connected to input.
        2) Remove parts of graph not connected to output.
        3) Reorder the vertices so that they are consecutive after steps 1 and 2.
        These 3 steps can be combined by deleting the rows and columns of the
        vertices that are not reachable from both the input and output (in reverse).
        """
        matrix = copy.deepcopy(original_matrix)
        ops = copy.deepcopy(original_ops)
        num_vertices = np.shape(original_matrix)[0]

        ops.insert(0, 'input')
        ops.append('output')

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if original_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if original_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            return matrix, ops, False

        matrix = np.delete(matrix, list(extraneous), axis=0)
        matrix = np.delete(matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del ops[index]

        # Infeasible Check
        if np.all(np.array(ops[1:-1]) == 4): # All operation is equal to skip-connection?
            return matrix, ops, False
        if len(ops) < 3:
            return matrix, ops, False
        if np.sum(matrix) > 9:
            return matrix, ops, False

        return matrix, ops, True
    
    def reset(self):
        for param in self.parameters():
            param.requires_grad = True
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)
                param.data.grad = None
    
    def encode(self, predictor_encoding, cutoff):

        if predictor_encoding == 'path':
            return encode_paths(self.get_path_indices())
        elif predictor_encoding == 'trunc_path':
            return encode_paths(self.get_path_indices())[:cutoff]
            

    def get_paths(self):
        """ 
        return all paths from input to output
        """
        ops = copy.deepcopy(self.org_ops)
        ops = [OPS[i] for i in ops]
        ops.insert(0, 'input')
        ops.append('output')

        paths = []
        for j in range(0, self.NUM_VERTICES):
            paths.append([[]]) if self.org_matrix[0][j] else paths.append([])
        
        # create paths sequentially
        for i in range(1, self.NUM_VERTICES - 1):
            for j in range(1, self.NUM_VERTICES):
                if self.org_matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, ops[i]])
        return paths[-1]

    def get_path_indices(self):
        """
        compute the index of each path
        There are 9^0 + ... + 9^5 paths total.
        (Paths can be length 0 to 5, and for each path, for each node, there
        are nine choices for the operation.)
        """
        paths = self.get_paths()
        ops = OPS
        mapping = {op: idx for idx, op in enumerate(OPS)}

        path_indices = []

        for path in paths:
            index = 0
            for i in range(self.NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(ops) ** i * (mapping[path[i]] + 1)

        path_indices.sort()
        return tuple(path_indices)
    
    def get_hash(self):
        return self.get_path_indices()
