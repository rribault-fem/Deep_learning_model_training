import torch.nn as nn
import numpy as np
from torchinfo import summary
import math

class conv1d_surr(nn.Module):
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
                  latent_space_dim:int=2**4, 
                  conv1DT_latent_dim : int= 2**9,
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
        self.conv1DT_latent_dim = conv1DT_latent_dim
        self.activation = activation
        if droupout_rate is not None:
            self.droupout =  nn.Dropout(p=droupout_rate)

        # find number of Layers :
        self.num_dense_layers = int(math.log(self.first_dense_layer_out_features, 2) - math.log(self.latent_space_dim, 2))+1
        self.num_conv1DT_layers = int(math.log(self.conv1DT_latent_dim, 2) - math.log(self.latent_space_dim, 2))
        self.num_conv1D_layers = int(math.log(self.conv1DT_latent_dim, 2) - math.log(self.spectrum_decomp_length, 2))

        of = first_dense_layer_out_features
        dense_layers_out_features = [int(of/(2**i)) for i in range(self.num_dense_layers-1)]
        dense_layers_out_features.append(self.latent_space_dim)
        of_list = dense_layers_out_features

        # Define the Dense layers of the neural network
        self.Dense1 = nn.Linear(self.x_input_size, of_list[0])
        for i in range(1, self.num_dense_layers) :
            dense_layer = nn.Linear(of_list[i-1], of_list[i])
            setattr(self, f"Dense{i+1}", dense_layer)

        # Define the Conv1DTranspose layers of the neural network
        def get_kernel_size(Lin, Lout, stride = 2) :
            kernel_size = Lout - stride *(Lin -1 )
            return kernel_size
        
        conv1DT_in_channel = [of_list[-1]*2**i for i in range(0, self.num_conv1DT_layers)]
        conv1DT_out_channel = [of_list[-1]*2**i for i in range(1, self.num_conv1DT_layers+1)]

        self.Conv1DT1 = nn.ConvTranspose1d(conv1DT_in_channel[0], conv1DT_out_channel[0], kernel_size=conv1DT_out_channel[0], stride=2 )
        for i in range(1, self.num_conv1DT_layers) :
            stride = 2
            kernel_size = get_kernel_size(conv1DT_in_channel[i], conv1DT_out_channel[i], stride = stride)
            conv1DT_layer = nn.ConvTranspose1d(conv1DT_in_channel[i], conv1DT_out_channel[i], kernel_size=kernel_size, stride=stride)
            setattr(self, f"Conv1DT{i+1}", conv1DT_layer)


        # finally filter the output to get the desired output shape
        conv1D_in_channel = [int(self.conv1DT_latent_dim/(2**i)) for i in range(self.num_conv1D_layers-1)]
        conv1D_out_channel = [int(self.conv1DT_latent_dim/(2**(i+1))) for i in range(self.num_conv1D_layers-1)]

        for i in range(self.num_conv1D_layers-1) :
            stride = 2
            conv1D_layer = nn.Conv1d(conv1D_in_channel[i], conv1D_out_channel[i], kernel_size=2, stride=stride )
            setattr(self, f"Conv1D{i}", conv1D_layer)
        
        stride = 2
        k_size = conv1D_out_channel[-1]-(self.spectrum_decomp_length-1)*stride
        conv1D_layer = nn.Conv1d(conv1D_out_channel[-1], self.spectrum_channel_nb, kernel_size=k_size, stride=stride )
        setattr(self, f"Conv1D{i+1}", conv1D_layer)


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
            x = self.activation(getattr(self, f"Dense{i+1}")(x))
            if hasattr(self, "dropout") :
                x = self.dropout(x)

        # Reshape encoded input to a 3D tensor with shape (None, 1, 16)
        x = x.view(-1, self.latent_space_dim, 1)

        # Decode the encoded input using Conv1DTranspose
        # first use wide kernel size to learn the global structure and interraction between channels
        for i in range(self.num_conv1DT_layers):
            x = self.activation(getattr(self, f"Conv1DT{i+1}")(x))
            if hasattr(self, "dropout") :
                x = self.dropout(x)
 
        # finally filter the output to get the desired output shape
        for i in range(self.num_conv1D_layers) :
            x = self.activation(getattr(self, f"Conv1D{i}")(x))
            if hasattr(self, "dropout") :
                x = self.dropout(x)
        x= x.permute(0, 2, 1)
        # for i in range(1):
        #     x = nn.functional.relu(self.Conv1D(x))
        return x


if __name__ == '__main__':
    # Test the model
    kwargs = {
        "x_input_size" : 7,
        "spectrum_decomp_length" : 512,
        "spectrum_channel_nb" : 18}

    model = conv1d_surr(first_dense_layer_out_features =2**7, 
    latent_space_dim =2**4,
    conv1DT_latent_dim = 2**9,
    **kwargs)