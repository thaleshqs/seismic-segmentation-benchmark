import numpy as np
import os
from torch.utils.data import Dataset


class SeismicDataset(Dataset):

    def __init__(self, data_path, orientation):
        self.data_path = data_path
        self.dataset_name = os.path.basename(self.data_path)

        self.data = np.load(os.path.join(self.data_path, f'{self.dataset_name}_dataset.npy'))
        self.labels = np.load(os.path.join(self.data_path, f'{self.dataset_name}_labels.npy'))

        self.n_inlines, self.n_crosslines, self.n_time_slices = self.data.shape
        self.orientation = orientation


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


    def get_class_weights(self):
        # INCOMPLETO!!!
        class_weights_dict = {
            'F3_alaudah'        : [0.7151, 0.8811, 0.5156, 0.9346, 0.9683, 0.9852],
            'F3_conoco_phillips': [],
            'F3_silva':           [],
            'parihaka':           [],
            'penobscot':          []
        }

        return class_weights_dict[self.dataset_name]
    

    def get_n_classes(self):
        # INCOMPLETO!!!
        n_classes_dict = {
            'F3_alaudah'        : 6,
            'F3_conoco_phillips': None,
            'F3_silva':           None,
            'parihaka':           None,
            'penobscot':          None
        }

        return n_classes_dict[self.dataset_name]
