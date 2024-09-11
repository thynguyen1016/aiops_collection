# evaluate.py

from sklearn import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

class ModelEvaluator:
    def __init__(self, y_train, y_pred, y_prob=None):
        """
        Initializes the evaluator with true labels and predicted labels.

        :param y_true: List or array of true labels
        :param y_pred: List or array of predicted labels
        :param y_prob: List or array of predicted probabilities (for AUC and ROC)
        """
        self.y_train = y_train
        self.y_pred = y_pred
        self.y_prob = y_prob

    def evaluate_accuracy(self):
        accuracy = accuracy_score(self.y_train, self.y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

    def evaluate_f1_score(self):
        f1 = f1_score(self.y_train, self.y_pred, average='weighted')  # 'weighted' to account for label imbalance
        print(f"F1 Score: {f1:.4f}")
        return f1

    def evaluate_auc(self):
        if self.y_prob is not None:
            auc = roc_auc_score(self.y_train, self.y_prob)
            print(f"AUC: {auc:.4f}")
            return auc
        else:
            print("AUC cannot be calculated without probability scores.")
            return None

    def evaluate_precision(self):
        precision = precision_score(self.y_train, self.y_pred, average='weighted')
        print(f"Precision: {precision:.4f}")
        return precision

    def evaluate_recall(self):
        recall = recall_score(self.y_train, self.y_pred, average='weighted')
        print(f"Recall: {recall:.4f}")
        return recall

    def confusion_matrix(self):
        cm = confusion_matrix(self.y_train, self.y_pred)
        print(f"Confusion Matrix: {cm}")
        return cm

    def evaluate_all(self):
        """
        Evaluate and print all metrics.
        """
        self.evaluate_accuracy()
        self.evaluate_f1_score()
        self.evaluate_precision()
        self.evaluate_recall()
        self.evaluate_auc()
        self.confusion_matrix()
