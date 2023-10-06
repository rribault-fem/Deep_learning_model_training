import torch.nn as nn
import numpy as np
from torchinfo import summary
import math

class conv1d_surr_nopca_ins2tens(nn.Module):
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
    def __init__(self, activation: nn.Module = nn.ReLU(),
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
            self.droupout =  nn.Dropout(p=droupout_rate)

        # Architecture of the neural network

        # Several conv1D layers are used to condense the input data per channels to a lattent space
        self.conv1 = nn.Conv2d(1125, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.minpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.dense1 = nn.Linear(16*32, 6)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, self.y_output_size)

        # The lattent space is transformed to a 1D tensor shape using a dense layer

        # find number of Layers :
        # self.num_dense_layers = 2
        # self.num_conv1D_layers = int(math.log(self.two_dims_decomp_length, 2) - math.log(self.latent_space_dim, 2))

        # dense_layers_out_features = []
        # dense_layers_out_features.append(self.latent_space_dim)
        # dense_layers_out_features.append(int(self.latent_space_dim/2))
        # dense_layers_out_features.append(self.y_output_size)
        # of_list = dense_layers_out_features

        # # Define the Dense layers of the neural network
        # self.Dense1 = nn.Linear(self.y_output_size, of_list[0])
        # for i in range(1, self.num_dense_layers+1) :
        #     dense_layer = nn.Linear(of_list[i-1], of_list[i])
        #     setattr(self, f"Dense{i+1}", dense_layer)

        # # Define the Conv1D layers of the neural network
        # def get_kernel_size(Lin, Lout, stride = 2) :
        #     kernel_size = Lout - stride *(Lin -1 )
        #     return kernel_size
        
        
        # conv1D_in_channel = [of_list[0]*2**i for i in range(0, self.num_conv1D_layers)]
        # conv1D_out_channel = [of_list[0]*2**i for i in range(1, self.num_conv1D_layers+1)]
        

        # self.Conv1D1 = nn.Conv1d(conv1D_in_channel[0], conv1D_out_channel[0], kernel_size=conv1D_out_channel[0], stride=2 )
        # for i in range(1, self.num_conv1D_layers) :
        #     stride = 2
        #     kernel_size = get_kernel_size(conv1D_in_channel[i], conv1D_out_channel[i], stride = stride)
        #     conv1D_layer = nn.Conv1d(conv1D_in_channel[i], conv1D_out_channel[i], kernel_size=kernel_size, stride=stride)
        #     if i==self.num_conv1D_layers-1 :
        #         conv1D_layer = nn.Conv1d(conv1D_in_channel[i], self.two_dims_channel_nb, kernel_size=kernel_size, stride=stride)
        #     setattr(self, f"Conv1D{i+1}", conv1D_layer)



        summary(self, input_size=(6,36000), gpu=True)     

    def forward(self, x):
        """
        Forward pass of the convolutional neural network.

        Args:
            x (torch.Tensor): Input data as a tensor of shape (batch_size, x_features_nb).

        Returns:
            torch.Tensor: A tensor representing the output of the convolutional neural network.
        """


        # reshape the input tensor in chuncks on the second dimension
        x = x.view((1125, 32 , self.two_dims_channel_nb))

        # use a conv2D each chunck to reduce dimensionality
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = x.view(16*32, 6)
        return x
    
if __name__ == '__main__':
    # Test the model
    kwargs = {
        "y_output_size" : 7,
        "two_dims_decomp_length" : 36000,
        "two_dims_channel_nb" : 6}

    model = conv1d_surr_nopca_ins2tens( 
    latent_space_dim =2**6,
    **kwargs)