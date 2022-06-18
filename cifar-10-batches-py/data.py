import numpy as np
import pickle


class DataLoader():
    def __init__(self, file) -> None:
        self.batch = pickle.load(open(file, 'rb'), encoding='bytes')
        self.batch_label = self.batch[b'batch_label']
        self.labels = np.array(self.batch[b'labels'])
        self.data = self.batch[b'data']
        self.filenames = self.batch[b'filenames']

batch = DataLoader("data_batch_5")
data, labels = batch.data, batch.labels
