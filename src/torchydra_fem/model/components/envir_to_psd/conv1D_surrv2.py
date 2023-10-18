import torch.nn as nn
import numpy as np
from torchinfo import summary
import math

class conv1d_surrv2(nn.Module):
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
                  first_dense_layer_out_features:int=2**7, 
                  latent_space_dim:int=512, 
                  droupout_rate : float= None, 
                  **kwargs):
        super().__init__()
        
        required_kwargs_list = ['x_input_size', 'spectrum_decomp_length', 'spectrum_channel_nb']
        for kwarg in required_kwargs_list:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required kwarg: {kwarg}")


        self.x_input_size : int = kwargs['x_input_size']
        self.spectrum_decomp_length : int = kwargs['spectrum_decomp_length']
        self.spectrum_channel_nb : int = kwargs['spectrum_channel_nb']
        self.first_dense_layer_out_features = first_dense_layer_out_features
        self.latent_space_dim = latent_space_dim
        self.activation = activation
        if droupout_rate is not None:
            self.droupout =  nn.Dropout(p=droupout_rate)

        # find number of Layers :
        self.num_dense_layers = int(math.log(self.latent_space_dim, 2) - math.log(self.first_dense_layer_out_features, 2))+1
        self.num_conv1d_layers = int(math.log(self.latent_space_dim, 2) - math.log(self.spectrum_decomp_length, 2))

        of = first_dense_layer_out_features
        dense_layers_out_features = [int(of*(2**i)) for i in range(self.num_dense_layers-1)]
        dense_layers_out_features.append(self.latent_space_dim)
        of_list = dense_layers_out_features

        # Define the dense layers of the neural network
        self.dense1 = nn.Linear(self.x_input_size, of_list[0])
        for i in range(1, self.num_dense_layers) :
            dense_layer = nn.Linear(of_list[i-1], of_list[i])
            setattr(self, f"dense{i+1}", dense_layer)

        # finally filter the output to get the desired output shape
        conv1d_in_channel = [1]
        conv1d_in_channel = conv1d_in_channel + [self.spectrum_channel_nb for i in range(self.num_conv1d_layers-1)]
        conv1d_out_channel = [self.spectrum_channel_nb for i in range(self.num_conv1d_layers)]

        for i in range(self.num_conv1d_layers) :
            stride = 2
            k_size = 2
            conv1d_layer = nn.Conv1d(conv1d_in_channel[i], conv1d_out_channel[i], kernel_size=k_size, stride=stride )
            setattr(self, f"conv1d{i}", conv1d_layer)


        summary(self, input_size=(self.x_input_size,))     

    def forward(self, x):
        """
        Forward pass of the convolutional neural network.

        Args:
            x (torch.Tensor): Input data as a tensor of shape (batch_size, x_features_nb).

        Returns:
            torch.Tensor: A tensor representing the output of the convolutional neural network.
        """
        # Encode the input using dense layers
        for i in range(self.num_dense_layers) :
            x = self.activation(getattr(self, f"dense{i+1}")(x))
            if hasattr(self, "dropout") :
                x = self.dropout(x)

        # Reshape encoded input to a 3D tensor with shape (None, 1, 16)
        x = x.unsqueeze(0)
        x = x.unsqueeze(-1)
        x = x.view(-1, 1, self.latent_space_dim)
 
        # finally filter the output to get the desired output shape
        for i in range(self.num_conv1d_layers) :
            x = self.activation(getattr(self, f"conv1d{i}")(x))
            if hasattr(self, "dropout") and i<self.num_conv1d_layers-2 :
                x = self.dropout(x)
        # flip channel and time axis
        x = x.permute(0,2,1)

        return x


if __name__ == '__main__':
    # Test the model
    kwargs = {
        "x_input_size" : 7,
        "spectrum_decomp_length" : 24,
        "spectrum_channel_nb" : 18}



    model = conv1d_surrv2(first_dense_layer_out_features =128, 
    latent_space_dim =512,
    **kwargs)