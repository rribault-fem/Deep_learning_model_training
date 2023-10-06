import torch
import numpy as np
from torchinfo import summary
import math

class SimpleIns2tens(torch.nn.Module):
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
    def __init__(self, activation: torch.nn.Module = torch.nn.ReLU(),
                  latent_space_dim:int=2**6, 
                  droupout_rate : float= None, 
                  **kwargs):
        super().__init__()
        
        required_kwargs_list = ['y_output_size', 'two_dims_decomp_length', 'two_dims_channel_nb']
        for kwarg in required_kwargs_list:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required kwarg: {kwarg}")


        self.y_output_size : int = kwargs['y_output_size']
        self.two_dims_decomp_length : int = kwargs['two_dims_decomp_length']
        self.two_dims_channel_nb : int = kwargs['two_dims_channel_nb']
        self.latent_space_dim = latent_space_dim
        self.activation = activation
        if droupout_rate is not None:
            self.droupout =  torch.nn.Dropout(p=droupout_rate)

        # Architecture of the neural network

        # Several conv1D layers are used to condense the input data per channels to a lattent space
        self.conv1 = torch.nn.Conv1d(3, 1, kernel_size=2, stride=1)
        
        # The lattent space is transformed to a 1D tensor shape using a dense layer

 

        summary(self, input_size=(32, 36000, 3))     

    def forward(self, x):
        """
        Forward pass of the convolutional neural network.

        Args:
            x (torch.Tensor): Input data as a tensor of shape (batch_size, x_features_nb).

        Returns:
            torch.Tensor: A tensor representing the output of the convolutional neural network.
        """


        # reshape the input tensor in chuncks on the second dimension
        x= x.permute(0, 2, 1)
        x = self.activation(self.conv1(x))

        _mean = torch.mean(x, dim=2, keepdim=True)
        _std = torch.std(x, dim=2, keepdim=True)
        _max = torch.max(x, dim=2, keepdim=True)[0]
        _min = torch.min(x, dim=2, keepdim=True)[0]

        # create a tensor with the statistics of x
        x = torch.cat((_mean, _std, _max, _min), dim=1)
        x = x.flatten(start_dim=1)
        
        # x = self.activation(self.conv5(x))
        # x = self.activation(self.conv6(x))
        # x = self.activation(self.conv7(x))
        # x = x.permute(0, 2, 1)
        # x = self.activation(self.dense1(x))
        # x = x.flatten(start_dim=1)
        # x = self.activation(self.dense1(x))
        # x = x.view(16*32, 6)
        
        
        return x
    
if __name__ == '__main__':
    # Test the model
    kwargs = {
        "y_output_size" : 7,
        "two_dims_decomp_length" : 36000,
        "two_dims_channel_nb" : 6}

    model = SimpleIns2tens( 
    latent_space_dim =2**6,
    **kwargs)