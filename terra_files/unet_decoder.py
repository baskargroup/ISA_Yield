import torch
from segmentation_models_pytorch.base.initialization import initialize_decoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torch import nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY
import pdb

@TERRATORCH_DECODER_REGISTRY.register
class UNetDecoder(nn.Module):
    """UNetDecoder. Wrapper around UNetDecoder from segmentation_models_pytorch to avoid ignoring the first layer."""

    def __init__(
        self, embed_dim: list[int], channels: list[int], use_batchnorm: bool = True, attention_type: str | None = None
    ):
        """Constructor

        Args:
            embed_dim (list[int]): Input embedding dimension for each input.
            channels (list[int]): Channels used in the decoder.
            use_batchnorm (bool, optional): Whether to use batchnorm. Defaults to True.
            attention_type (str | None, optional): Attention type to use. Defaults to None
        """
        if len(embed_dim) != len(channels):
            msg = "channels should have the same length as embed_dim"
            raise ValueError(msg)
        super().__init__()
        # pdb.set_trace()
        # embed_dim = [1536, 3072, 6144, 6144] # ← Anirudha's default code
        embed_dim = [2304, 4608, 9216, 9216]  # 12 timesteps - chloe
        # embed_dim = [4608, 9216, 18432, 18432]  # ← 24 timesteps - chloe
        self.decoder = UnetDecoder(
            encoder_channels=[embed_dim[0], *embed_dim],
            decoder_channels=channels,
            n_blocks=len(channels),
            use_norm=use_batchnorm,
            add_center_block=False,
            attention_type=attention_type,
        )
        initialize_decoder(self.decoder)
        self.out_channels = channels[-1]

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        # The first layer is ignored in the original UnetDecoder, so we need to duplicate the first layer
        x = [x[0].clone(), *x]
        return self.decoder(x)
