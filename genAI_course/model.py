
class AI:
    def __init__(self, algorithm, model_type):
        self.algorithm = algorithm
        self.model_type = model_type

    def fit(self, model, X_train, y_train):
        print(f"Fitting {self.algorithm} model with {self.model_type} type on data.")
        print(f"Model: {model}")
        print(f"X_train: {X_train}")
        print(f"y_train: {y_train}")

    def predict(self, model, X_pred):
        print(f"Predicting using {self.algorithm} model.")
        print(f"Model: {model}")
        print(f"X_pred: {X_pred}")


class DeepLearning(AI):
    def __init__(self, algorithm, model_type):
        super().__init__(algorithm, model_type)

    def fit(self, model, X_train, y_train, learning_rate):
        print(f"Fitting {self.algorithm} model with learning rate {learning_rate}.")
        print(f"Model: {model}")
        print(f"X_train: {X_train}")
        print(f"y_train: {y_train}")

    def train_on_epoch(self, model, X_train, y_train, epoch):
        print(f"Training {self.algorithm} model on epoch {epoch}.")
        print(f"Model: {model}")
        print(f"X_train: {X_train}")
        print(f"y_train: {y_train}")
