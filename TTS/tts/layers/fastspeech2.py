"""FastSpeech2 encoder,decoder, variation controller and attention."""

import torch
import math
import numpy as np
from collections import OrderedDict
from typing import Union, Optional, Sequence
from .tacotron2 import ConvBNBlock
from TTS.tts.utils.generic_utils import pad_list, generate_masks_mel_length

class ConvLNBlock(ConvBNBlock):
    """
        A 1d convolution with Layer normalisation and relu activation
    """
    def __init__(self,
                input_channels: int,
                out_channels: int, 
                kernel_size: int,
                activation: Union[None, str]="relu",
                drop_out: float=0.5):
        """
            ConvLNBlock constructor.
        """
        super(ConvLNBlock, self).__init__(input_channels, out_channels, kernel_size, activation, drop_out)
        self.normalization = torch.nn.LayerNorm(out_channels)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.convolution1d(x)
        x = self.activation(x)
        x = x.permute(0,2,1)
        x = self.normalization(x)
        x = self.dropout(x)
        return x

class MultiheadedAttention(torch.nn.Module):
    """
        Multihead self attention module based on `Attention is all you need transformer` 
    """
    def __init__(self,
                embedding_dims: int,
                n_heads: int=4,
                dim_k: int=64,
                dim_v: int=64,
                **kwargs):
        """
            Multihead self attention module based on `Attention is all you need transformer`
        """
        super(MultiheadedAttention, self).__init__()
        self.embedding_dims = embedding_dims
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self._mask_value = 1e-8
        self.normalising_term = torch.sqrt(torch.tensor(self.dim_k,dtype=torch.float32))

        self.query_embeddings = torch.nn.Linear(self.embedding_dims, self.n_heads*self.dim_k)
        self.key_embeddings = torch.nn.Linear(self.embedding_dims, self.n_heads*self.dim_k)
        self.values_embeddings = torch.nn.Linear(self.embedding_dims, self.n_heads*self.dim_v)
        self.sm = torch.nn.Softmax(dim=2)
        self.linear_projection = torch.nn.Linear(self.n_heads*self.dim_v, self.embedding_dims)
        
        self.layer_norm = torch.nn.LayerNorm(self.embedding_dims)

        self.alignment_score = None

    def cal_query_key_value(self, input):
        """
            Calculte query,key and value vectors for a given input
        """
        query = self.query_embeddings(input)
        key = self.key_embeddings(input)
        value = self.values_embeddings(input)
        return query, key, value

    def forward(self, batch, mask=None):
        q, k, v = self.cal_query_key_value(batch)

        # Calculate score for attention
        self.alignment_score = torch.bmm(q, k.permute(0,2,1)) / self.normalising_term
        if not mask == None:
            self.alignment_score = self.alignment_score.masked_fill_(~mask.unsqueeze(-1), 0)
            #print (self.alignment_score.squeeze().size())
            attention = self.sm(self.alignment_score.squeeze())
        else:
            attention = self.sm(self.alignment_score)

        context = torch.bmm(attention, v)
        context = self.linear_projection(context)

        return self.layer_norm(context+batch), attention



class Convd1d2Layers(torch.nn.Module):
    """
        A class for two layered conv1d as mentioned in fastspeech paper
    """
    def __init__(self,
                input_channels,
                con1_out_channels,
                h_param_dict):
        super(Convd1d2Layers, self).__init__()
        self.input_channels = input_channels
        self.con1_out_channels = con1_out_channels

        self.conv1 = ConvBNBlock(self.input_channels, self.con1_out_channels, kernel_size=h_param_dict['fft_conv1d_kernel_size'][0], activation="relu")
        self.conv2 = ConvBNBlock(self.con1_out_channels, self.input_channels, kernel_size=h_param_dict['fft_conv1d_kernel_size'][1], activation=None)

        self.layer_norm = torch.nn.LayerNorm(self.input_channels)

    def forward(self, batch):
        residual = batch.permute(0,2,1)
        residual = self.conv1(residual)
        residual = self.conv2(residual)
        residual = residual.permute(0,2,1)
        return self.layer_norm(residual+batch)

class FFT(torch.nn.Module):
    """
        Feed forward transformer block with a feed forward network and self attention layer
    """
    def __init__(self,
                embedding_dims: int,
                conv1_output_channels: int,
                hparams_dict: dict,
                dim_k: int = 64,
                dim_v: int = 64,
                n_heads: int = 4,
                ):
        super(FFT, self).__init__()
        self.multihead_attention = MultiheadedAttention(embedding_dims, n_heads=n_heads, dim_k=dim_k, dim_v=dim_v)
        self.ff_layer = Convd1d2Layers(embedding_dims, conv1_output_channels, hparams_dict)

    def forward(self, batch, mask=None):
        context, self_alignment = self.multihead_attention(batch, mask)
        if not mask == None:
            context = context.masked_fill_(~mask.unsqueeze(-1), 0.0)
        encoder_output = self.ff_layer(context)
        if not mask == None:
            encoder_output = encoder_output.masked_fill_(~mask.unsqueeze(-1), 0.0)
        return encoder_output, self_alignment

class ScaledPositionalEncoding(torch.nn.Module):
    """
        Class for calculating learned scaled embeddings
    """
    def __init__(self):
        super(ScaledPositionalEncoding, self).__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def reset_scale(self):
        self.scale = torch.tensor(1.0)

    def forward(self,batch):
        num_inp_tokens = batch.size(1)
        num_channels = batch.size(2)
        
        position_encodings = batch.new(num_inp_tokens, num_channels)

        positions_inp_tokens = torch.arange(0,num_inp_tokens, dtype=torch.float32)
        position_channeles = torch.arange(0,num_channels, dtype=torch.float32)
        traingular_position_info = positions_inp_tokens.unsqueeze(1) * torch.pow(100000,2*position_channeles/num_channels).unsqueeze(0)

        position_encodings[:, 0::2] = torch.sin(traingular_position_info[:, 0::2])
        position_encodings[:, 1::2] = torch.cos(traingular_position_info[:, 1::2])

        encoded_batch = batch + self.scale*position_encodings.unsqueeze(0)
        return encoded_batch


class Transformer(torch.nn.Module):
    """
        Transformer based on transformer TTS
    """
    def __init__(self,
                embedding_dims: int,
                conv1_output_channels: int,
                hparams_dict: dict,
                dim_k: int = 64,
                dim_v: int = 64,
                n_heads: int = 4,
                num_fft_block: int = 5,
                r: int = 1
            ):
        super(Transformer, self).__init__()

        self.embedding_dims = embedding_dims
        self.num_fft_block = num_fft_block
        self.r = 1
        
        # Select a positional encoding layer
        self.pos_enc_class = ScaledPositionalEncoding()
        self.fft_layers = torch.nn.ModuleList([FFT(self.embedding_dims, conv1_output_channels, hparams_dict,
                                                                        dim_k = dim_k, dim_v = dim_v, n_heads = n_heads)
                                                                        for _ in range(self.num_fft_block)])
    def forward(self, batch, mask=None, return_attention=True):
        encoder_outputs = self.pos_enc_class(batch)
        encoder_self_attention = []

        for fft_block in self.fft_layers:
            encoder_outputs, self_attention = fft_block(encoder_outputs, mask)
            if return_attention:
                encoder_self_attention += [self_attention]

        return encoder_outputs, encoder_self_attention


class VaraincePredictor(torch.nn.Module):
    """
        Base variance adaptor class for pitch, energy and length predictors
    """
    def __init__(self,
                 input_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 drop_out: int,
                 n_layers: int = 2,
                ):
        super(VaraincePredictor, self).__init__()
        assert n_layers >= 2, "Keep number of convolution layers in variance predictor more or equal to than 2, as described in the paper"

        var_predictor_modules = []
        var_predictor_modules.append(ConvLNBlock(input_channels, out_channels, kernel_size, drop_out=drop_out))
        for _ in range(n_layers-1):
            var_predictor_modules.append(ConvLNBlock(out_channels, out_channels, kernel_size, drop_out=drop_out))
        var_predictor_modules.append(torch.nn.Linear(out_channels, 1))

        self.predictor = torch.nn.Sequential(*var_predictor_modules)

    def forward(self, batch: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        batch = self.predictor(batch)
        if not mask == None:
            batch = batch.masked_fill_(~mask.unsqueeze(-1), 0)
        return batch

class DurationRegulator(torch.nn.Module):
    """
        Duration regulator as mentioned in the fastspeech/ fastspeech2 paper
    """
    def __init__(self):
        super(DurationRegulator, self).__init__()

    def forward(self, batch: torch.Tensor,token_lengths: torch.Tensor, token_durations: torch.Tensor, alpha: int = 1.0) -> Sequence[torch.Tensor]:
        if alpha != 1.0:
            token_durations = torch.round(token_durations.float() * alpha)

        token_list = [token[:len_token] for token, len_token in zip(batch, token_lengths)]
        dur_list   = [token_duration[:len_token] for token_duration, len_token in zip(token_durations, token_lengths)]

        expanded_durs = [self.inflate(tokens, durs) for tokens, durs in zip(token_list, dur_list)]
        padded_batch = pad_list(expanded_durs)
        mel_lens = torch.Tensor([regulated_mel.size(0) for regulated_mel in expanded_durs]).to(padded_batch.device)
        return padded_batch, mel_lens

    def inflate(self, tokens: torch.Tensor, durs: torch.Tensor) -> torch.Tensor:
        """
            Takes an input token tensor and their durations and expand token tensor
        """
        if durs.sum() == 0:
            print ("all of the predicted durations are 0. fill 0 with 1.")
            durs = durs.fill_(1)
        inflated_tensor = [token.unsqueeze(0).repeat_interleave(math.ceil(dur), dim=0)
                            for token, dur in zip(tokens, durs) if dur > 0]
        if not inflated_tensor:
            return tokens[0].new_zeros(size=(1,tokens[0].size()[-1]))
        inflated_tensor = torch.cat(inflated_tensor, dim=0)
        return inflated_tensor


class DurationPredictor(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 drop_out: int,
                 n_layers: int = 2
                ):
        super(DurationPredictor, self).__init__()
        self.duraion_predictor = VaraincePredictor(input_channels, out_channels, kernel_size, drop_out, n_layers)
        self.duration_regulator = DurationRegulator()

    def forward(self, 
                batch: torch.Tensor,
                token_lengths: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                label_durations: Optional[torch.Tensor] = None,
                alpha: int = 1
               ) -> torch.Tensor:
        predicted_durations = self.duraion_predictor(batch, mask)
        if not label_durations == None:
            padded_batch, mel_lengths = self.duration_regulator(batch, token_lengths, label_durations, alpha=alpha)
            return padded_batch, mel_lengths, predicted_durations
        padded_batch, mel_lengths = self.duration_regulator(batch, token_lengths, predicted_durations, alpha=alpha)
        return padded_batch, mel_lengths, predicted_durations



class VarianceAdaptor(torch.nn.Module):
    """
        Variance adaptor as described in Fastspeech2 paper
    """
    def __init__(self,
                 input_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 pitch_bank_param_dict: dict,
                 energy_bank_param_dict: dict,
                 drop_out: float = 0.5,
                 n_layers: int = 2,
                ):
        super(VarianceAdaptor, self).__init__()

        self.duration_pred_regulator = DurationPredictor(input_channels, out_channels, kernel_size, drop_out, n_layers=2)
        self.pitch_predictor = VaraincePredictor(input_channels, out_channels, kernel_size, drop_out, n_layers=2)
        self.energy_predictor = VaraincePredictor(input_channels, out_channels, kernel_size, drop_out, n_layers=2)


        # Enegry and pitch banks are from fastpitch and espnet implementation of fastspeech2
        pitch_bank_kernel_size = pitch_bank_param_dict["kernel_size"]
        pitch_bank_padding = (pitch_bank_kernel_size-1)//2
        pitch_bank_drop_out = pitch_bank_param_dict["drop_out"]
        self.pitch_bank  = torch.nn.Sequential(
                                                OrderedDict(
                                                    [
                                                        ("conv1", torch.nn.Conv1d(in_channels=1,out_channels=input_channels,
                                                                                kernel_size=pitch_bank_kernel_size, padding=pitch_bank_padding)),
                                                        ("drop_out", torch.nn.Dropout(p=pitch_bank_drop_out))
                                                    ]
                                                )
                                              )
        energy_bank_kernel_size = energy_bank_param_dict["kernel_size"]
        energy_bank_padding = (energy_bank_kernel_size-1)//2
        energy_bank_drop_out = energy_bank_param_dict["drop_out"]
        self.energy_bank  = torch.nn.Sequential(
                                                    OrderedDict(
                                                        [                                                       
                                                            ("conv1", torch.nn.Conv1d(in_channels=1,out_channels=input_channels,
                                                                                    kernel_size=energy_bank_kernel_size,padding=energy_bank_padding)),
                                                            ("drop_out", torch.nn.Dropout(p=energy_bank_drop_out))
                                                        ]
                                                    )
                                                )

    def forward(self, batch:torch.Tensor, token_lengths:torch.Tensor,
                    label_durations: Optional[torch.Tensor]=None, label_pitch: Optional[torch.Tensor]=None, label_energy: Optional[torch.Tensor]=None,
                    input_mask: Optional[torch.Tensor]=None) -> Sequence[torch.Tensor]:
            
            # Calculate duration and regulated batch
            batch, mel_lengths, duration_p = self.duration_pred_regulator(batch, token_lengths, label_durations = label_durations, mask=input_mask)
            mel_mask = generate_masks_mel_length(mel_lengths)
            pitch_p  = self.pitch_predictor(batch, mask=mel_mask)
            energy_p = self.energy_predictor(batch, mask=mel_mask)

            if not label_energy == None:
                energy_embeddings = self.energy_bank(label_energy.permute(0,2,1))
                batch += energy_embeddings.permute(0,2,1)
            else:
                energy_embeddings = self.energy_bank(energy_p.permute(0,2,1))
                batch += energy_embeddings.permute(0,2,1)

            if not label_pitch == None:
                pitch_embeddings = self.pitch_bank(label_pitch.permute(0,2,1))
                batch += pitch_embeddings.permute(0,2,1)
            else:
                pitch_embeddings = self.pitch_bank(pitch_p.permute(0,2,1))
                batch += pitch_embeddings.permute(0,2,1)

            return batch, duration_p, pitch_p, energy_p, mel_mask

    def inference(self, batch:torch.Tensor, token_lengths:torch.Tensor, alpha_pitch=1.0, alpha_energy=1.0, alpha_speed=1.0) -> Sequence[torch.Tensor]:
            batch, mel_lengths, duration_p = self.duration_pred_regulator(batch, token_lengths, alpha=alpha_speed)
            mel_mask = generate_masks_mel_length(mel_lengths)

            pitch_p  = self.pitch_predictor(batch, mask=mel_mask)
            energy_p = self.energy_predictor(batch, mask=mel_mask)

            pitch_embeddings = self.pitch_bank(pitch_p.permute(0,2,1))*alpha_pitch
            energy_embeddings = self.energy_bank(energy_p.permute(0,2,1))*alpha_energy

            batch += pitch_embeddings.permute(0,2,1) + energy_embeddings.permute(0,2,1)

            return batch, duration_p, pitch_p, energy_p
    

class PostNet(torch.nn.Module):
    """
        Post net class
    """
    def __init__(self):
        """
            Postnet class constructor
        """
        raise NotImplemented()