import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RADFE(nn.Module):
    def __init__(self, num_features=192, hidden_dim=1024, num_layers=1, linear_dims=[4, 8], conv_channels = 64):
        super(RADFE, self).__init__()
        channels = 16
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Linear layers before GRU
        self.linears = nn.ModuleList()
        last_dim = num_features
        for next_dim in linear_dims:
            self.linears.append(nn.Linear(last_dim * channels, next_dim * channels))
            last_dim = next_dim
        
        last_dim = last_dim * channels


        # GRU layer
        self.gru = nn.GRU(last_dim, self.hidden_dim, num_layers, batch_first=True)

       

        self.reshape_size = int((64 *self.hidden_dim) ** 0.5)
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(1, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1)

        # Upsampling layers
        self.upsample = nn.Upsample(size=(126, 224), mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(conv_channels, 1, kernel_size=1)  # Output layer


    def forward(self, x):
       
        
        x_t = x.view(x.shape[0], x.shape[1], -1) 
        # print("x_t_in", x_t.shape)
        for linear in self.linears:
            x_t = F.relu(linear(x_t))
            # print("x_t_Linear", x_t.shape)
        out, h = self.gru(x_t)
        # print("out", out.shape)
        lstm_concat = out
        x_t = lstm_concat.reshape(lstm_concat.shape[0], -1)
        x_t = x_t.view(-1,1, self.reshape_size, self.reshape_size)
        # print("x_t", x_t.shape)
        x_t = F.relu(self.conv1(x_t))
        x_t = F.relu(self.conv2(x_t))
        x_t = F.relu(self.conv3(x_t))
        x_t = self.upsample(x_t)
        x_t = self.conv4(x_t)
 
        x_t = x_t.view(x_t.shape[0], 1, 126, 224)
        return x_t


# Test case
def test_radfe_model():
    num_features = 192
    hidden_dim = 1024
    num_layers = 1
    linear_dims = [8, 16]
    conv_channels = 32
    batch_size = 1
  

    # Initialize the model
    model = RADFE(num_features, hidden_dim, num_layers, linear_dims, conv_channels)
    print(model)
    # Generate random input tensor
    # The input dimensions should match the expected input of the first linear layer after being reshaped
    input_tensor = torch.randn(batch_size, 64, 16, 192)

    # Forward pass
    output = model(input_tensor)

    # Print the output shape
    print("Output shape:", output.shape)

# Run the test case

if(__name__ == "__main__"):
    test_radfe_model()