import torch.nn as nn
import numpy as np
from torchinfo import summary
import math

class conv1d_autoencoder(nn.Module):
    """
    A PyTorch Lightning module representing a 1D convolutional neural network
    for surrogate modeling of data.

    Args:
        spectrum_channel_nb (int): Number of channels in the output spectrum.
        x_features_nb (int): Number of features in the input data.

    Returns:
        torch.Tensor: A tensor representing the output of the convolutional neural network.
    """
    def __init__(self, activation: nn.Module = nn.ReLU(),
                  latent_space_dim:int=2**4, 
                  droupout_rate : float= None, 
                  **kwargs):
        super().__init__()
        
        required_kwargs_list = ['spectrum_decomp_length', 'spectrum_channel_nb']
        for kwarg in required_kwargs_list:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required kwarg: {kwarg}")


        self.spectrum_decomp_length : int = kwargs['spectrum_decomp_length']
        self.spectrum_channel_nb : int = kwargs['spectrum_channel_nb']
        self.latent_space_dim = latent_space_dim
        self.activation = activation
        if droupout_rate is not None:
            self.droupout =  nn.Dropout(p=droupout_rate)

        # find number of Layers :
        self.num_conv1DT_layers = int(math.log(self.spectrum_decomp_length, 2) - math.log(self.latent_space_dim, 2))
        self.num_conv1D_layers = int(math.log(self.spectrum_decomp_length, 2) - math.log(self.latent_space_dim, 2))

        # Encode the spectrum to get the desired output shape

        # Decode usingConv1DTranspose layers of the neural network
   

    def forward(self, y):
        """
        Forward pass of the convolutional neural network.

        Args:
            x (torch.Tensor): Input data as a tensor of shape (batch_size, x_features_nb).

        Returns:
            torch.Tensor: A tensor representing the output of the convolutional neural network.
        """
        # Encode the input 
        # decode the input
        x=y
        return x


if __name__ == '__main__':
    # Test the model
    kwargs = {
        "spectrum_decomp_length" : 512,
        "spectrum_channel_nb" : 18}

    model = conv1d_autoencoder(
    latent_space_dim =2**4,
    **kwargs)