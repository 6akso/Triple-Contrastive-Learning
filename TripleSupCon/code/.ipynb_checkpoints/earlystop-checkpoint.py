class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
class EarlyStopping:  

    def __init__(self, patience=30, min_delta=0):  

        """  

        :param patience: 多少个epoch后，如果准确率没有改进则停止  

        :param min_delta: 准确率的最小改进量，低于此值则不视为改进  

        """  
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
            # 重置计数器如果验证准确率提高了  
            self.counter = 0  
        elif val_accuracy - self.best_accuracy < 0:  # 注意这里使用负号，因为我们希望准确率增加  
            self.counter += 1  
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")  
            if self.counter >= self.patience:  
                print('INFO: Early stopping')  
                self.early_stop = True  
