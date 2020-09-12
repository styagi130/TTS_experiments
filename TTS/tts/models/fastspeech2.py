from typing import Sequence, Optional
import torch
from TTS.tts.layers.fastspeech2 import Transformer, VarianceAdaptor, PostnetResidual
from TTS.tts.utils.generic_utils import sequence_mask

class Fastspeech2(torch.nn.Module):
    """
        Base class for fastSpeech2
    """
    def __init__(self, num_inputs: int, num_out_mels: int, num_input_channels: int, encoder_kernel_size: int, decoder_kernel_size: int,
                variance_adaptor_kernel_size: int, pitch_hparams: dict, energy_hparams: int, use_postnet: bool = False):
        super(Fastspeech2, self).__init__()
        self.symbol_embeddings = torch.nn.Embedding(num_inputs, num_input_channels)

        self.encoder = Transformer(num_input_channels, num_input_channels, {"fft_conv1d_kernel_size":[encoder_kernel_size, encoder_kernel_size]},
                                    num_fft_block=4, n_heads=2, dim_k=64, dim_v=64)
        self.variation_adaptor = VarianceAdaptor(num_input_channels, num_input_channels, variance_adaptor_kernel_size, pitch_hparams, energy_hparams)
        self.decoder = Transformer(num_input_channels, num_input_channels, {"fft_conv1d_kernel_size":[decoder_kernel_size, decoder_kernel_size]},
                                    num_fft_block=4, n_heads=2, dim_k=64, dim_v=64)
        
        self.linear_projection = torch.nn.Linear(num_input_channels, num_out_mels)

        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostnetResidual(num_out_mels)

    def forward(self, batch: torch.Tensor, input_lengths: torch.Tensor, 
                    label_durations: Optional[torch.Tensor] = None, label_pitch: Optional[torch.Tensor] = None, label_energy: Optional[torch.Tensor] = None,
                    input_mask: Optional[torch.Tensor] = None) -> Sequence[torch.Tensor]:
        if input_mask is None:
            input_mask = self.compute_mask(input_lengths)
        batch = self.symbol_embeddings(batch)
        batch, encoder_alignments = self.encoder(batch, input_mask)
        batch, duration_p, pitch_p, energy_p, decoder_mask = self.variation_adaptor(batch, input_lengths, 
                                                                                label_durations = label_durations, label_pitch = label_pitch,
                                                                                label_energy = label_energy,
                                                                                input_mask = input_mask)
        batch, decoder_alignments = self.decoder(batch, decoder_mask)
        mels = self.linear_projection(batch)
        if self.use_postnet:
            mels_post = mels.permute(0,2,1)
            mels_post = self.postnet(mels_post)
            mels_post = mels_post.permute(0,2,1)
            return mels_post, mels, duration_p, pitch_p, energy_p, encoder_alignments, decoder_alignments
        return mels, mels, duration_p, pitch_p, energy_p, encoder_alignments, decoder_alignments

    def inference(self, text,input_lengths=None, speaker_ids=None, alpha_pitch=1.0, alpha_energy=1.0, alpha_speed=1.2):
        if input_lengths is None:
            input_lengths = torch.IntTensor([text.size(1)])
        print (input_lengths, text.size())
        embedded_inputs = self.symbol_embeddings(text)
        batch, _ = self.encoder(embedded_inputs)
        batch, duration_p, pitch_p, energy_p = self.variation_adaptor.inference(batch, input_lengths, alpha_pitch=alpha_pitch, alpha_energy=alpha_energy, alpha_speed=alpha_speed)
        batch, _ = self.decoder(batch)
        mels = self.linear_projection(batch)
        if self.use_postnet:
            mels_post = mels.permute(0,2,1)
            mels_post = self.postnet(mels_post)
            mels_post = mels_post.permute(0,2,1)
            return mels_post.detach(), mels.detach(), duration_p.detach(), (pitch_p.detach(), energy_p.detach())
        return mels.detach(), mels.detach(), duration_p.detach(), (pitch_p.detach(), energy_p.detach())

    def compute_mask(self, input_lengths):
        device = input_lengths.device
        input_mask = sequence_mask(input_lengths).to(device)
        return input_mask