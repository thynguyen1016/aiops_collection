# train.py

from model import DeepLearning
from dataset import Dataset

if __name__ == '__main__':
    # Tạo dữ liệu giả
    X_train = [[1, 2], [3, 4], [5, 6]]
    y_train = [0, 1, 0]
    X_pred = [[7, 8], [9, 10]]

    # Khởi tạo object Dataset
    dataset = Dataset(X_train, y_train)
    X, y = dataset.get_data()

    # Khởi tạo object DeepLearning
    dl_model = DeepLearning("Neural Network", "classification")

    # Sử dụng phương thức fit
    dl_model.fit(model="NN_Model", X_train=X, y_train=y, learning_rate=0.001)

    # Sử dụng phương thức predict
    dl_model.predict(model="NN_Model", X_pred=X_pred)

    # Sử dụng phương thức train_on_epoch
    dl_model.train_on_epoch(model="NN_Model", X_train=X, y_train=y, epoch=10)
