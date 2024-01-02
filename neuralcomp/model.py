import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from vector_quantize_pytorch import VectorQuantize

class RunningMode(Enum):
    encoder_decoder = 1
    encoder = 2
    decoder = 3

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self._initialize_layers()
        self.running_mode = RunningMode.encoder_decoder

    def _initialize_layers(self):
        """ Initialize the layers of the autoencoder. """
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(in_channels=64, out_channels=16, kernel_size=16, stride=8, padding=0)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.ConvTranspose1d(in_channels=16, out_channels=64, kernel_size=16, stride=8, padding=0),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        ])

    def _apply_activation(self, x, layer, activation_fn=F.relu):
        """ Apply a convolution layer and an activation function. """
        return activation_fn(layer(x))

    def encoder(self, x):
        """ Encoder function. """
        for layer in self.encoder_layers[:-1]:
            x = self._apply_activation(x, layer)
        return F.leaky_relu(self.encoder_layers[-1](x))

    def decoder(self, y):
        """ Decoder function. """
        for layer in self.decoder_layers[:-1]:
            y = self._apply_activation(y, layer)
        return nn.LeakyReLU(0.1)(self.decoder_layers[-1](y))

    def forward(self, x):
        """ Forward pass through the model. """
        if self.running_mode == RunningMode.encoder_decoder:
            return self.decoder(self.encoder(x))
        elif self.running_mode == RunningMode.encoder:
            return self.encoder(x)
        else:  # RunningMode.decoder
            return self.decoder(x)

    def display(self):
        """ Display layer information. """
        for layer in self.children():
            print(f"Layer : {layer}")
            print("Parameters : ")
            for param in layer.parameters():
                print(param.shape)

    def set_mode(self, mode: RunningMode):
        """ Set the running mode of the autoencoder. """
        assert isinstance(mode, RunningMode), "Mode must be an instance of RunningMode enum."
        self.running_mode = mode
