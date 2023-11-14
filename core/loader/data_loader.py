import numpy as np
import os
import json
from torch.utils.data import Dataset


class SeismicDataset(Dataset):

    def __init__(self, data_path, orientation, compute_weights=False, remove_faulty_slices=True):
        self.data_path = data_path
        self.dataset_name = os.path.basename(self.data_path)
        self.faulty_slices_path = os.path.join(self.data_path, 'faulty_slices.json')

        self.data = np.load(os.path.join(self.data_path, f'{self.dataset_name}_dataset.npy'))
        self.labels = np.load(os.path.join(self.data_path, f'{self.dataset_name}_labels.npy'))

        if remove_faulty_slices:
            self.__remove_faulty_slices()
        
        self.n_classes = self.__process_class_labels()

        self.orientation = orientation
        self.n_inlines, self.n_crosslines, self.n_time_slices = self.data.shape
        self.weights = self.__compute_class_weights() if compute_weights else None


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
    

    def __compute_class_weights(self):
        total_n_values = self.n_inlines * self.n_crosslines * self.n_time_slices
        # Weights are inversely proportional to the frequency of the classes in the training set
        _, counts = np.unique(self.labels, return_counts=True)
        
        return total_n_values / (counts*self.n_classes)
    

    def __remove_faulty_slices(self):
        try:
            with open(self.faulty_slices_path, 'r') as json_buffer:
                # File containing the list of slices to delete
                faulty_slices = json.loads(json_buffer.read())

                self.data = np.delete(self.data, obj=faulty_slices['inlines'], axis=0)
                self.data = np.delete(self.data, obj=faulty_slices['crosslines'], axis=1)
                self.data = np.delete(self.data, obj=faulty_slices['time_slices'], axis=2)

                self.labels = np.delete(self.labels, obj=faulty_slices['inlines'], axis=0)
                self.labels = np.delete(self.labels, obj=faulty_slices['crosslines'], axis=1)
                self.labels = np.delete(self.labels, obj=faulty_slices['time_slices'], axis=2)

        except FileNotFoundError:
            print('"Remove faulty slices" is on, but no file with the indices was found.')
            print('Training with the whole volume instead.\n')

            pass
    

    def __process_class_labels(self):
        # Labels must be in the range [0, number_of_classes) for the loss function to work properly
        label_values = np.unique(self.labels)
        new_labels_dict = {label_values[i]: i for i in range(len(label_values))}

        for key, value in zip(new_labels_dict.keys(), new_labels_dict.values()):
            self.labels[self.labels == key] = value
        
        return len(label_values)
    

    def get_class_weights(self):
        return self.weights
    

    def get_n_classes(self):
        return self.n_classes
