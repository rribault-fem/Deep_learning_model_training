import torch
import numpy as np
from torchinfo import summary
import math

class TimeSeriesToTimeSeriesDense(torch.nn.Module):
    """
    A PyTorch Lightning module representing a 1D convolutional neural network
    for regression from ins time series to statistics of tension sensor.

    the input is a 2D tensor of shape (batch_size, x_channels_nb, spectrum_decomp_length)
    the output is a 1D tensor of shape (batch_size, y_output_size)

    A MLP is used to reduce the number of features from x_channels_nb to x_features_nb

    Args:
        spectrum_channel_nb (int): Number of channels in the input time series condensed with PCA.
        x_features_nb (int): Number of features in the input data.

    Returns:
        torch.Tensor: A tensor representing the output of the convolutional neural network.
    """
    def __init__(self,
                dropout_rate : float= None,
                **kwargs):
        super().__init__()
        
        required_kwargs_list = ['two_dims_decomp_length', 'two_dims_channel_nb']
        for kwarg in required_kwargs_list:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required kwarg: {kwarg}")
        
        self.nb_obs : int = kwargs['nb_obs']
        self.two_dims_decomp_length : int = kwargs['two_dims_decomp_length']
        self.two_dims_channel_nb : int = kwargs['two_dims_channel_nb']

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        if dropout_rate is not None:
            self.dropout =  torch.nn.Dropout(p=dropout_rate)


        # Architecture of the neural network

        # Several conv1D layers are used to condense the input data per channels to a lattent space
        self.dense1 = torch.nn.Linear(self.two_dims_channel_nb, 128)
        self.dense2 = torch.nn.Linear(128, 256)
        self.dense3 = torch.nn.Linear(256, 1)
        
        # The lattent space is transformed to a 1D tensor shape using a dense layer

 

        summary(self, input_size=(self.two_dims_decomp_length, self.two_dims_channel_nb))     

    def forward(self, x):
        """
        Forward pass of the convolutional neural network.

        Args:
            x (torch.Tensor): Input data as a tensor of shape (batch_size, x_features_nb).

        Returns:
            torch.Tensor: A tensor representing the output of the convolutional neural network.
        """
        # reshape the input tensor in chuncks on the second dimension
        # x = x.view(self.nb_obs*self.two_dims_decomp_length, self.two_dims_channel_nb)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.dense3(x)
        
        return x
    
if __name__ == '__main__':
    # Test the model
    kwargs = {
        "nb_obs" : 30,
        "two_dims_decomp_length" : 600,
        "two_dims_channel_nb" : 3}

    model = TimeSeriesToTimeSeriesConv1D(dropout_rate=0.35, 
    **kwargs)