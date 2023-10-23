import numpy as np
import os
from torch.utils.data import Dataset


class SeismicDataset(Dataset):

    def __init__(self, data_path, orientation, compute_weights=False):
        self.data_path = data_path
        self.dataset_name = os.path.basename(self.data_path)

        self.data = np.load(os.path.join(self.data_path, f'{self.dataset_name}_dataset.npy'))
        self.labels = np.load(os.path.join(self.data_path, f'{self.dataset_name}_labels.npy'))

        self.n_inlines, self.n_crosslines, self.n_time_slices = self.data.shape
        self.orientation = orientation
        self.weights = None

        # If weighted loss is enabled
        if compute_weights:
            self.weights = self.compute_class_weights()


    def __getitem__(self, index):
        if self.orientation == 'in':
            image = self.data[index, :, :]
            label = self.labels[index, :, :]
        else:
            image = self.data[:, index, :]
            label = self.labels[:, index, :]
        
        # Reshaping to 3D image
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        return image, label


    def __len__(self):
        return self.n_inlines if self.orientation == 'in' else self.n_crosslines
    

    def compute_class_weights(self):
        total_n_values = self.n_inlines*self.n_crosslines*self.n_time_slices

        _, counts = np.unique(self.labels, return_counts=True)
        counts = 1 - (counts/total_n_values)

        return counts
    

    def get_class_weights(self):
        return self.weights
    

    def get_n_classes(self):
        return len(self.weights)
