from torch.nn import Conv1d, ConvTranspose1d,  Module, Dropout, ReLU, GELU, LeakyReLU, BatchNorm1d, MaxPool1d, LSTM
from torchinfo import summary
import torch

class TimeSeriesToTimeSeriesConv1D(Module):
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
                latent_space_dim:int=2**8, 
                dropout_rate : float= 0.1, 
                activation : str = 'nn.GELU', 
                **kwargs):
        super().__init__()
        
        required_kwargs_list = ['two_dims_decomp_length', 'two_dims_channel_nb', 'nb_obs']
        for kwarg in required_kwargs_list:
            if kwarg not in kwargs:
                raise ValueError(f"Missing required kwarg: {kwarg}")
            
        if dropout_rate is not None:
            self.dropout =  Dropout(p=dropout_rate)

        activation_dict = {
            'nn.ReLU' : ReLU(),
            'nn.LeakyReLU' : LeakyReLU(),
            'nn.GELU' : GELU()
        }
        
        self.activ = activation_dict[activation]
        self.latent_space_dim = latent_space_dim

        self.nb_obs : int = kwargs['nb_obs']
        self.two_dims_decomp_length : int = kwargs['two_dims_decomp_length']
        self.two_dims_channel_nb : int = kwargs['two_dims_channel_nb']

        self.activ = ReLU()


        # Architecture of the neural network

        # Several conv1D layers are used to condense the input data per channels to a lattent space
        self.conv1 = Conv1d(3, 32, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv1d(32, self.latent_space_dim, kernel_size=1, stride=1, padding=0)
        self.conv3 = Conv1d(self.latent_space_dim, 1, kernel_size=1, stride=1, padding=0)

        # The lattent space is transformed to a 1D tensor shape using a dense layer
        # self.convT1 = ConvTranspose1d(1, 1, kernel_size=2, stride=2, padding=0)
        # self.convT2 = ConvTranspose1d(1, 1, kernel_size=2, stride=2, padding=0)
        # self.convT3 = ConvTranspose1d(1, 1, kernel_size=2, stride=2, padding=0)

        # self.LSTM = LSTM(input_size=2392, hidden_size=5, num_layers=1, batch_first=True)

        # self.conv_chan = Conv1d(self.two_dims_channel_nb, 1, kernel_size=3, stride=1, padding=1)

        summary(self, input_size=(self.nb_obs, self.two_dims_decomp_length, self.two_dims_channel_nb))     

    def forward(self, x):
        """
        Forward pass of the convolutional neural network.

        Args:
            x (torch.Tensor): Input data as a tensor of shape (batch_size, x_features_nb).

        Returns:
            torch.Tensor: A tensor representing the output of the convolutional neural network.
        """
        # reshape the input tensor in chuncks on the second dimension
        #x = x.view(self.nb_obs*self.two_dims_decomp_length, self.two_dims_channel_nb)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        # x = self.MaxPool1d(x)
        # x = BatchNorm1d(512)(x)
        x = self.dropout(self.activ(x))
        x = self.conv2(x)
        # x = self.MaxPool1d(x)
        # x = BatchNorm1d(128)(x)
        x = self.dropout(self.activ(x))
        x = self.conv3(x)
        # x = self.MaxPool1d(x)
        x = self.activ(x)


        # x = x.permute(0, 2, 1)
        # x = self.conv_chan(x)
        # x = x.permute(0, 2, 1)

        # x = self.convT1(x)
        # # x = BatchNorm1d(128)(x)
        # x = self.dropout(self.activ(x))
        # x = self.convT2(x)
        # # x = BatchNorm1d(512)(x)
        # x = self.dropout(self.activ(x))
        # x = self.convT3(x)
        # x = self.activ(x)
        # x_last= x.squeeze(1)
        # x_last, _ = self.LSTM(x_last)
        # x_last = x_last.unsqueeze(1)
        # x = torch.concatenate((x, x_last), dim=2)
        x = x.permute(0, 2, 1)


        return x
    
if __name__ == '__main__':
    # Test the model
    kwargs = {
        "nb_obs" : 30,
        "two_dims_decomp_length" : 2399,
        "two_dims_channel_nb" : 3}

    model = TimeSeriesToTimeSeriesConv1D( 
    **kwargs)