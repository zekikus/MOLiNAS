class EarlyStopping:


    def __init__(self, patience=0):
        """
        This function implements early stopping during training based on the validation loss and score.
        
        :param patience: The number of epochs to wait before stopping training if the validation loss
        does not improve. If the validation loss does not improve for `patience` consecutive epochs,
        training will stop, defaults to 0 (optional)
        """
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_loss = None
        self.best_score = None
        self.worst_counter = 0
    
    def stopTraining(self, epoch, valid_loss, valid_score):

        if epoch == 0:
            self.best_loss = valid_loss
            self.best_score = valid_score

        if self.best_loss < valid_loss:
            self.worst_counter = self.worst_counter + 1
        else:
            self.worst_counter = 0
            self.best_loss = valid_loss
        
        if valid_score > self.best_score:
            self.best_score = valid_score

        if self.worst_counter >= self.patience:
            self.stopped_epoch = epoch
            return True
        else:
            return False