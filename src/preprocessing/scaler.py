from dataclasses import dataclass
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
import logging
import sklearn.preprocessing as sklpre
from sklearn.base import TransformerMixin
import os

@dataclass
class Scaler(ABC) :


    @abstractmethod
    def scale_data(self, numpy_channels_training_set : np.array, numpy_channels_test_set: np.array) ->  Tuple[np.array, np.array] :
        """
                This method scales the data using the specified scaler by self.scaler_option.
        """
    @abstractmethod
    def scale_data_infer(self, numpy_channels_set : np.array) ->  np.array:
        """
        This method scales the data using the specified scaler by self.scaler_option.
        """


@dataclass
class Scaler_1D(Scaler) :
    
    scaler_option : str
    scalers : Optional[List] = None
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
        log.info(f'Using sklearn scaler: {self.scaler_option}')

        self.scalers = []

        if hasattr(sklpre, self.scaler_option):
                    self.scalers.append(getattr(sklpre, self.scaler_option)())
                    channels_training_set = self.scalers[0].fit_transform(channels_training_set)
                    channels_val_set = self.scalers[0].transform(channels_test_set)

        else:
            raise ValueError(f"No such method in sklearn.preprocessing: {self.scaler_option}")
        
        log.info('1D data scaled')
        log.info('###')

        return channels_training_set, channels_val_set
    
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
        return self.scalers[0].inverse_transform(Channels_set)
    

@dataclass
class Scaler_2D(Scaler) :
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

        # Perform scaling on each channel
        for i in range(n_channels):
            slc_train = Y_numpy_channels_training_set[:, :, i]
            slc_test = Y_numpy_channels_test_set[:, :, i]
            
            # Select user defined scalling from hydra config file
            if hasattr(sklpre, self.scaler_option):
                scaler = getattr(sklpre, self.scaler_option)() 
                slc_scaled_train = scaler.fit_transform(slc_train)
                slc_scaled_test = scaler.transform(slc_test)
            else:
                raise ValueError(f"No such method in sklearn.preprocessing: {self.scaler_option}")

            # Store the decomposition
            y_train[:, :, i] = slc_scaled_train
            y_test[:, :, i] = slc_scaled_test
            self.scalers.append(scaler) # store the decompositions for later use in the inverse transform
           
        return y_train, y_test
    
    def scale_data_infer(self, Y_numpy_channels_set : np.array) ->  np.array:
        """
        This method scales the data using the specified scaler by self.scaler_option.

        Args:
            Y_numpy_channels_training_set (np.array): A numpy array containing the training set data.

        Returns:
            tuple: y_set.
        """
        
        log =  logging.getLogger(os.environ['logger_name'])
        log.info('scale the 2D data')
        
        # Prepare empty output arrays
        n_samples_infer = np.shape(Y_numpy_channels_set)[0]
        spectrum_length = np.shape(Y_numpy_channels_set)[1]
        n_channels = np.shape(Y_numpy_channels_set)[2]
        y_infer = np.zeros((n_samples_infer, spectrum_length, n_channels))

        log.info(f'Using sklearn scaler: {self.scaler_option}')

        # Perform scalling on each channel
        for i in range(n_channels):
            
            if hasattr(sklpre, self.scaler_option):    
                slc_infer = Y_numpy_channels_set[:, :, i]        
                slc_scaled_infer = self.scalers[i].transform(slc_infer)
                
            else:
                raise ValueError(f"No such method in sklearn.preprocessing scalling: {self.scaler_option}")

            # Store the decomposition
            y_infer[:, :, i] = slc_scaled_infer
               
        return y_infer
    
    def inverse_transform_data_infer(self, Y_numpy_channels_set : np.array) ->  np.array:
        """
        This method scales the data using the specified scaler by self.scaler_option.

        Args:
            Y_numpy_channels_training_set (np.array): A numpy array containing the training set data.

        Returns:
            tuple: y_set.
        """
        
        log =  logging.getLogger(os.environ['logger_name'])
        log.info('scale the 2D data')
        
        # Prepare empty output arrays
        n_samples_infer = np.shape(Y_numpy_channels_set)[0]
        spectrum_length = np.shape(Y_numpy_channels_set)[1]
        n_channels = np.shape(Y_numpy_channels_set)[2]
        y_infer = np.zeros((n_samples_infer, spectrum_length, n_channels))

        log.info(f'Using sklearn scaler: {self.scaler_option}')

        # Perform scalling on each channel
        for i in range(n_channels):
            
            if hasattr(sklpre, self.scaler_option):    
                slc_infer = Y_numpy_channels_set[:, :, i]        
                slc_scaled_infer = self.scalers[i].inverse_transform(slc_infer)
                
            else:
                raise ValueError(f"No such method in sklearn.preprocessing scalling: {self.scaler_option}")

            # Store the decomposition
            y_infer[:, :, i] = slc_scaled_infer
               
        return y_infer

