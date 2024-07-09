from torch.nn import Conv1d, ConvTranspose1d,  Module, Dropout, ReLU, GELU, LeakyReLU, BatchNorm1d, Linear
from torchinfo import summary

class TimeSeriesToStatsConv1d(Module):
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
                latent_space_dim:int=2**6, 
                dropout_rate : float= 0.3, 
                activation : str = 'nn.GELU',
                active_dense : bool = True,
                batch_norm : bool = True, 
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
        self.active_dense = active_dense
        self.batch_norm = batch_norm
        self.activ = activation_dict[activation]
        self.latent_space_dim = latent_space_dim
        self.nb_obs : int = kwargs['nb_obs']
        self.two_dims_decomp_length : int = kwargs['two_dims_decomp_length']
        self.two_dims_channel_nb : int = kwargs['two_dims_channel_nb']


        # Architecture of the neural network

        # Several conv1D layers are used to condense the input data per channels to a lattent space
        self.Conv1d_1 = Conv1d(6, self.latent_space_dim, kernel_size=1, stride=1, padding=0)
        self.Conv1d_2 = Conv1d(self.latent_space_dim, self.latent_space_dim*2, kernel_size=1, stride=1, padding=0)
        self.Conv1d_3 = Conv1d(self.latent_space_dim*2, self.latent_space_dim, kernel_size=1, stride=1, padding=0)
        self.Conv1d_4 = Conv1d(self.two_dims_decomp_length, self.latent_space_dim, kernel_size=1, stride=1, padding=0)
        
        self.BatchNorm1d1 =  BatchNorm1d(self.latent_space_dim)
        self.BatchNorm1d2 = BatchNorm1d(self.latent_space_dim*2)
        self.BatchNorm1d3 = BatchNorm1d(self.latent_space_dim)

        # The convolutionnal layers are followed by a dense layer
        self.dense1 = Linear(self.latent_space_dim, 1)
        self.dense2 = Linear(self.latent_space_dim, 4)


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
        x = self.Conv1d_1(x)
        if self.batch_norm :
            x = self.BatchNorm1d1(x)
        x = self.dropout(self.activ(x))

        x = self.Conv1d_2(x)
        if self.batch_norm :
            x = self.BatchNorm1d2(x)
        x = self.dropout(self.activ(x))

        x = self.Conv1d_3(x)
        if self.batch_norm :
            x = self.BatchNorm1d3(x)
        x = self.dropout(self.activ(x))

        x = x.permute(0, 2, 1)
        x = self.Conv1d_4(x)
        if self.batch_norm :
            x = self.BatchNorm1d3(x)
        x = self.dropout(self.activ(x))

        x = self.dense1(x)
        if self.active_dense :
            x = self.activ(x)
        x = x.permute(0, 2, 1)
        x = self.dense2(x)
        if self.active_dense :
            x = self.activ(x)
        x = x.permute(0, 2, 1)
        
        return x
    
if __name__ == '__main__':
    # Test the model
    kwargs = {
        "nb_obs" : 3270,
        "two_dims_decomp_length" : 600,
        "two_dims_channel_nb" : 6}

    model = TimeSeriesToStatsConv1d( 
    **kwargs)