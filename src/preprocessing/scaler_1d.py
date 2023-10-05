import logging
import sklearn.preprocessing as sklpre
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import os
from scaler import Scaler


@dataclass
class Scaler_1D(Scaler) :
    
    scaler : TransformerMixin 
    # the name of a method defined below. This name is defined in Hydra config file
    # can be any sklearn transformer or a custom transformer with transformer API

    def scale_data(self, channels_training_set : np.array, channels_test_set: np.array) -> Tuple[np.array, np.array]:
        """
        This method scales the data using the specified scaler.

        Args:
            X_numpy_channels_training_set (np.array): A numpy array containing the training set data.
            X_numpy_channels_test_set (np.array): A numpy array containing the testing set data.

        Returns:
            tuple: A tuple containing two elements: x_numpy_channels_training_set and x_numpy_channels_val_set.
        """
        
        log =  logging.getLogger(os.environ['logger_name'])
        # Select user defined Scaler
        log.info('###')
        log.info("select environmental values scaler")

        # Fit and transform the data
        x_numpy_channels_training_set = self.scaler.fit_transform(channels_training_set)
        x_numpy_channels_val_set = self.scaler.transform(channels_test_set)
        
        log.info('1D data scaled')
        log.info('###')

        return x_numpy_channels_training_set, x_numpy_channels_val_set
    
    def scale_data_infer(self, Channels_set : np.array) ->  np.array:
        """
        This method scales the data using the specified scaler by self.scaler.

        Args:
            Channels_set (np.array): A numpy array containing the set data.

        Returns:
            np.array : channels_set.
        """
        
        log =  logging.getLogger(os.environ['logger_name'])
        log.info('scale the 1D data')
        return self.scaler.inverse_transform(Channels_set)
    
