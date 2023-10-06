import logging
from dataclasses import dataclass
from typing import List
import numpy as np
import sklearn.decomposition as skldecomp
from typing import List, Optional, Tuple
import os

@dataclass
class Decomp_y_spectrum :
    """ This class allow the user to select pca methods for the y data"""
    
    # The pca option is the name of the method to use for the pca transformation
    decomp_option : str
    # The number of component to use for the pca
    decomp_n_component : int
    decomps : Optional[List] = None

    def decomp_data(self, y_numpy_channels_training_set : np.array, y_numpy_channels_test_set: np.array) -> Tuple[np.array, np.array]:
        """ This function perform the pca on the y data"""
        log =logging.getLogger(os.environ['logger_name'])
        log.info('decomposition of the y spectrum data')
        
        # Prepare empty output arrays
        n_samples_train = np.shape(y_numpy_channels_training_set)[0]
        n_samples_test = np.shape(y_numpy_channels_test_set)[0]
        n_channels = np.shape(y_numpy_channels_training_set)[2]
        y_train_decomp = np.zeros((n_samples_train, self.decomp_n_component, n_channels))
        y_test_decomp = np.zeros((n_samples_test, self.decomp_n_component, n_channels))
        self.decomps = []

        log.info(f'Using sklearn decomposition: {self.decomp_option}')
        # Perform decomposition on each channel
        for i in range(n_channels):
            slc_train = y_numpy_channels_training_set[:, :, i]
            slc_test = y_numpy_channels_test_set[:, :, i]
            
            # Select user defined decomposition from hydra config file
            if hasattr(skldecomp, self.decomp_option):
                decomp = getattr(skldecomp, self.decomp_option)(n_components= self.decomp_n_component)
            
                slc_decomp_train = decomp.fit_transform(slc_train)
                log.info(f"Explained variance ratio for channel {i}: {np.sum(decomp.explained_variance_ratio_)}")
                slc_decomp_test = decomp.transform(slc_test)
            else:
                raise ValueError(f"No such method in skleanr decomposition: {self.decomp_option}")

            # Store the decomposition
            y_train_decomp[:, :, i] = slc_decomp_train
            y_test_decomp[:, :, i] = slc_decomp_test
            self.decomps.append(decomp) # store the decompositions for later use in the inverse transform
        
        log.info('decomposition of the y spectrum data done')
        log.info('###')
        
        return y_train_decomp, y_test_decomp