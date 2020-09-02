from typing import Sequence, Optional
import torch
from TTS.tts.layers.fastspeech2 import Transformer, VarianceAdaptor, PostNet

class FastSpeech2(torch.nn.Module):
    """
        Base class for fastSpeech2
    """
    def __init__(self, num_inputs: int, num_out_mels: int, num_input_channels: int, encoder_kernel_size: int, decoder_kernel_size: int,
                variance_adaptor_kernel_size: int, pitch_hparams: dict, energy_hparams: int, use_postnet: bool = False):
        super(FastSpeech2, self).__init__()
        self.character_embeddings = torch.nn.Embedding(num_inputs, num_input_channels)

        self.encoder = Transformer(num_input_channels, num_input_channels, {"fft_conv1d_kernel_size":[encoder_kernel_size, encoder_kernel_size]},
                                    num_fft_block=4, n_heads=4, dim_k=64, dim_v=64)
        self.variation_adaptor = VarianceAdaptor(num_input_channels, num_input_channels, variance_adaptor_kernel_size, pitch_hparams, energy_hparams)
        self.decoder = Transformer(num_input_channels, num_input_channels, {"fft_conv1d_kernel_size":[decoder_kernel_size, decoder_kernel_size]},
                                    num_fft_block=4, n_heads=4, dim_k=64, dim_v=64)
        
        self.linear_projection = torch.nn.Linear(num_input_channels, num_out_mels)

        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()

    def forward(self, batch: torch.Tensor, input_lengths: torch.Tensor, 
                    label_durations: Optional[torch.Tensor] = None, label_pitch: Optional[torch.Tensor] = None, label_energy: Optional[torch.Tensor] = None,
                    input_mask: Optional[torch.Tensor] = None) -> Sequence[torch.Tensor]:
        batch = self.character_embeddings(batch)
        batch, encoder_alignments = self.encoder(batch, input_mask)
        batch, duration_p, pitch_p, energy_p, decoder_mask = self.variation_adaptor(batch, input_lengths, 
                                                                                label_durations = label_durations, label_pitch = label_pitch,
                                                                                label_energy = label_energy,
                                                                                input_mask = input_mask)
        batch, decoder_alignments = self.decoder(batch, decoder_mask)
        mels = self.linear_projection(batch)
        if self.use_postnet:
            mels_post = self.postnet(mels)
            return mels_post, mels, duration_p, pitch_p, energy_p, encoder_alignments, decoder_alignments
        return mels, mels, duration_p, pitch_p, energy_p, encoder_alignments, decoder_alignments