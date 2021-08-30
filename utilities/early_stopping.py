class EarlyStopping():
    def __init__(self, mode, patience):
        """
        :param mode: min or max
        :param patience: nof epochs to wait before stopping
        """
        self.mode = mode
        self.patience = patience
        self.current_patience = patience
        if self.mode == "max":
            self.best_metric = 0.0
        else:
            self.best_metric = 10000.0

    def stop(self, current_metric):
        if self.mode == "max":
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.current_patience = self.patience
            else:
                self.current_patience -= 1

            print("patience left:{}, best({})".format(self.current_patience, self.best_metric))
            print()

            if self.current_patience == 0:
                return True
            else:
                return False
        else:
            # mode = "min"
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.current_patience = self.patience
            else:
                self.current_patience -= 1

            print("patience left:{}, best({})".format(self.current_patience, self.best_metric))
            print()

            if self.current_patience == 0:
                return True
            else:
                return False


