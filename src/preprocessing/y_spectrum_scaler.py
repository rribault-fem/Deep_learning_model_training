import logging
from dataclasses import dataclass
from typing import List
import numpy as np
import sklearn.preprocessing as sklpre
from typing import List, Optional, Tuple
import os

@dataclass
class Y_spectrum_scaler :
    """ This class allow the user to select pca methods for the y data"""
    
    # The pca option is the name of the method to use for the pca transformation
    scaler_option : str
    scalers : Optional[List] = None

    def scale_data(self, Y_numpy_channels_training_set : np.array, Y_numpy_channels_test_set: np.array) ->  Tuple[np.array, np.array]:
        """
        This method scales the data using the specified scaler by self.scaler_option.

        Args:
            Y_numpy_channels_training_set (np.array): A numpy array containing the training set data.
            Y_numpy_channels_test_set (np.array): A numpy array containing the testing set data.

        Returns:
            tuple: A tuple containing two elements: y_train and y_test.
        """
        
        log =  logging.getLogger(os.environ['logger_name'])
        log.info('scale the 2D data')
        
        # Prepare empty output arrays
        n_samples_train = np.shape(Y_numpy_channels_training_set)[0]
        n_samples_test = np.shape(Y_numpy_channels_test_set)[0]
        spectrum_length = np.shape(Y_numpy_channels_training_set)[1]
        n_channels = np.shape(Y_numpy_channels_training_set)[2]
        y_train = np.zeros((n_samples_train, spectrum_length, n_channels))
        y_test = np.zeros((n_samples_test, spectrum_length, n_channels))
        self.scalers = []

        log.info(f'Using sklearn scaler: {self.scaler_option}')

        # Perform decomposition on each channel
        for i in range(n_channels):
            slc_train = Y_numpy_channels_training_set[:, :, i]
            slc_test = Y_numpy_channels_test_set[:, :, i]
            
            # Select user defined decomposition from hydra config file
            if hasattr(sklpre, self.scaler_option):
                scaler = getattr(sklpre, self.scaler_option)() 
                slc_scaled_train = scaler.fit_transform(slc_train)
                slc_scaled_test = scaler.transform(slc_test)
            else:
                raise ValueError(f"No such method in skleanr decomposition: {self.decomp_option}")

            # Store the decomposition
            y_train[:, :, i] = slc_scaled_train
            y_test[:, :, i] = slc_scaled_test
            self.scalers.append(scaler) # store the decompositions for later use in the inverse transform
           
        return y_train, y_test