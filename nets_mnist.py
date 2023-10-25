"""Proposed residual neural nets architectures suited for MNIST"""

from typing import List

import torch
import torch.nn as nn

class BaseEncoder(nn.Module):
    """This is a base class for Encoders neural networks."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own encoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseEncoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Encoder(BaseEncoder):
            ...
            ...     def __init__(self):
            ...         BaseEncoder.__init__(self)
            ...         # your code
            ...
            ...     def forward(self, x: torch.Tensor):
            ...         # your code
            ...         output = ModelOutput(
            ...             embedding=embedding,
            ...             log_covariance=log_var # for VAE based models
            ...         )
            ...         return output

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder
        """
        raise NotImplementedError()


class BaseDecoder(nn.Module):
    """This is a base class for Decoders neural networks."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, z: torch.Tensor):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own decoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseDecoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_decoder(BaseDecoder):
            ...
            ...    def __init__(self):
            ...        BaseDecoder.__init__(self)
            ...        # your code
            ...
            ...    def forward(self, z: torch.Tensor):
            ...        # your code
            ...        output = ModelOutput(
            ...             reconstruction=reconstruction
            ...         )
            ...        return output

        Parameters:
            z (torch.Tensor): The latent data that must be decoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the decoder

        .. note::

            By convention, the reconstruction tensors should be in [0, 1] and of shape
            BATCH x channels x ...

        """
        raise NotImplementedError()




class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return x + self.conv_block(x)
    
class Encoder(BaseEncoder):
    def __init__(self):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 64, 64)
        self.latent_dim = 64
        self.n_channels = 1
        self.trial = 'ciao'

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )
        layers.append(nn.ReLU(), nn.Linear(128 * 4 * 4, 64))
        self.layers = layers
        self.depth = len(layers)

    

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""
     

        max_depth = self.depth

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

        print(out.shape)

        return out




class Decoder(BaseDecoder):
   
    def __init__(self):
        BaseDecoder.__init__(self)

        self.input_dim = (1, 64, 64)
        self.latent_dim = 64
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(nn.Linear(64, 128 * 4 * 4))

        layers.append(nn.ConvTranspose2d(128, 128, 3, 2, padding=1))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )

        layers.append(
            nn.Sequential(nn.ConvTranspose2d(128, 128, 5, 2, padding=1), nn.Sigmoid())
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 5, 2, padding=1, output_padding=1)
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, self.n_channels, 4, 2, padding=1), nn.Sigmoid()
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):

        max_depth = self.depth

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)
            if i == 0:
                out = out.reshape(z.shape[0], 128, 4, 4)

        return out

