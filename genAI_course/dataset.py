
class Dataset:
    def __init__(self, input_datas, labels):
        self.input_datas = input_datas
        self.labels = labels

    def get_data(self):
        print(f"Input Data: {self.input_datas}")
        print(f"Labels: {self.labels}")
        return self.input_datas, self.labels