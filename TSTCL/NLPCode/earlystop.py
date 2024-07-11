class EarlyStopping:

    def __init__(self, patience=10, min_delta=0):
        """   """
        self.patience = patience
        self.min_delta = 0
        self.counter = 0
        self.best_accuracy = None
        self.early_stop = False
    def __call__(self, val_accuracy):
        if self.best_accuracy is None:
            self.best_accuracy = val_accuracy
        elif val_accuracy - self.best_accuracy > 0:
            self.best_accuracy = val_accuracy
            self.counter = 0
        elif val_accuracy - self.best_accuracy < 0:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
