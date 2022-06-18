import numpy as np
import pickle

class DataLoader():
    def __init__(self, file) -> None:
        self.batch = pickle.load(open(file, 'rb'), encoding='bytes')
        self.name = self.batch[b'batch_label']
        self.labels = np.array(self.batch[b'labels'])
        self.data = self.batch[b'data']
        self.files = self.batch[b'filenames']

batch = DataLoader("train5")
data, labels, name, files = batch.data, batch.labels, batch.name, batch.files
print(name)