"""Tests for individual layer of fastspeech2"""
import sys
import torch
import numpy as np

sys.path.append("./..")
from TTS.tts.layers.fastspeech2 import MultiheadedAttention, Convd1d2Layers, FFT, Transformer, VarianceAdaptor
from TTS.tts.layers.losses import Fastspeech2Loss
from TTS.tts.models.fastspeech2 import Fastspeech2
from TTS.utils.io import AttrDict

def gen_batch(batch_size, num_tokens, embedding_dim):
    return torch.randn(size=(batch_size, num_tokens, embedding_dim))

def gen_batch_mask(batch_size, num_tokens = 10):
    num_tokens = 10
    sequence_lengths = torch.randint(low=num_tokens//2, high=num_tokens, size=(batch_size,))
    num_tokens = sequence_lengths.max().item()
    batch = torch.randint(low=0,high=20,size=(batch_size, num_tokens))

    seq_range = torch.arange(0,num_tokens).long()
    seq_range_expanded = seq_range.unsqueeze(0).expand(batch_size, num_tokens)
    sequence_lengths_expanded = sequence_lengths.unsqueeze(1).expand_as(seq_range_expanded)
    return batch, sequence_lengths, seq_range_expanded < sequence_lengths_expanded

def test_MultiheadedAttention(batch,do_mask=False):
    args_dict = {
                "embedding_dims":batch.shape[-1]
               }
    kwargs_dict = {
                    "n_heads":4,
                    "dim_k":64,
                    "dim_v":128
                 }
    multiheaded_attention = MultiheadedAttention(*args_dict.values(),**kwargs_dict)
    
    sequence_lengths, mask = None, None
    if do_mask:
        sequence_lengths, mask = gen_batch_mask(batch)
    context, alignments = multiheaded_attention(batch,mask)

    print ("\n##############SequenceLe#############\n")
    print (sequence_lengths)
    print ("\n##############Alignments#############\n")
    print (alignments)
    print ("\n##############Encocder_o#############\n")
    print (context)

def test_Convd1d2Layers(batch):
    args_dict = {
                "embedding_dims":batch.shape[-1],
                "out_channels":batch.shape[-1]*2,
                "hparams_dict": {"fft_conv1d_kernel_size":[3, 3]}
               }
    ff = Convd1d2Layers(*args_dict.values())
    ff_outs = ff(batch)

    print ("\n##############FF Outs#############\n")
    print (ff_outs)
    print ("\n##############Shapes############\n")
    print (f"ff_outs: {ff_outs.shape} ~~~~ Input Batch: {batch.shape}")

def test_FFT(batch,do_mask=False):
    sequence_lengths, mask = None, None
    if do_mask:
        sequence_lengths, mask = gen_batch_mask(batch)
    args_dict = {
                "embedding_dims":batch.shape[-1],
                "out_channels":batch.shape[-1]*2,
                "hparams_dict": {"fft_conv1d_kernel_size":[3, 3]}
                }
    kwargs_dict = {
                    "n_heads":4,
                    "dim_k":64,
                    "dim_v":128
                  }
    fft_block = FFT(*args_dict.values(),**kwargs_dict)
    encoder_outputs, alignments = fft_block(batch,mask)

    print ("\n##############SequenceLe#############\n")
    print (sequence_lengths)
    print ("\n##############Alignments#############\n")
    print (alignments)
    print ("\n##############Encocder_o#############\n")
    print (encoder_outputs)

def test_Transformer(batch,do_mask=False):
    sequence_lengths, mask = None, None
    if do_mask:
        sequence_lengths, mask = gen_batch_mask(batch)
    args_dict = {
                "embedding_dims":batch.shape[-1],
                "out_channels":batch.shape[-1]*2,
                "hparams_dict": {"fft_conv1d_kernel_size":[3, 3]}
                }
    kwargs_dict = {
                    "num_fft_block": 5,
                    "n_heads": 4,
                    "dim_k": 64,
                    "dim_v": 128
                  }
    fft_block = Transformer(*args_dict.values(),**kwargs_dict)
    encoder_outputs, alignments = fft_block(batch, mask)

    print ("\n##############SequenceLe#############\n")
    print (sequence_lengths)
    print ("\n##############Alignments#############\n")
    print (alignments[-1])
    print ("\n##############Encocder_o#############\n")
    print (encoder_outputs)

def test_variance_adaptor(batch, do_mask=False):
    sequence_lengths, mask = None, None
    if do_mask:
        sequence_lengths, mask = gen_batch_mask(batch)
    args_dict = {
                    "input_channels": batch.size(-1),
                    "output_channels": batch.size(-1),
                    "kernel_size": 3,
                    "pitch_bank_param_dict": {"kernel_size":3,"drop_out":0.5},
                    "energy_bank_param_dict": {"kernel_size":3,"drop_out":0.5}
                }
    kwargs_dict = {
                  }
    v_a = VarianceAdaptor(*args_dict.values(), **kwargs_dict)
    print ("\n##############SequenceLe#############\n")
    print (sequence_lengths)

    ## Test inferencing, i.e no true duration, pitch, energy
    v_a.eval()
    batch_pred, duration_p, pitch_p, energy_p, mel_mask = v_a(batch, sequence_lengths, input_mask=mask)
    print ("\n##############VA Inference Tests#############\n")
    print (batch_pred.shape, duration_p.shape, pitch_p.shape, energy_p.shape, mel_mask.shape)

    # Test training, i.e given duration, pitch, energy
    v_a.train()
    durations = torch.randint(low=2, high=6, size=batch.size()[:-1])

    duration_max = [duration[:idx].sum() for duration, idx in zip(durations, sequence_lengths)]
    durations_max = max(duration_max)

    pitch = torch.randn(size=(batch_size, durations_max)).unsqueeze(-1)
    energy = torch.randn(size=(batch_size, durations_max)).unsqueeze(-1)
    batch, duration_p, pitch_p, energy_p, mel_mask = v_a(batch, sequence_lengths, 
                                                        label_durations=durations, label_pitch=pitch, label_energy=energy)

    print ("\n##############VA Train Tests#############\n")
    print (batch.shape, duration_p.shape, pitch_p.shape, energy_p.shape, mel_mask.shape)
    

def test_fastspeech2(batch, do_mask):
    sequence_lengths, mask = None, None
    batch, sequence_lengths, mask = gen_batch_mask(batch_size)
    args_dict = {
                    "num_inputs": 40,
                    "num_out_mels": 80,
                    "num_input_channels": batch.size(-1),
                    "encoder_kernel_size": 3,
                    "decoder_kernel_size": 3,
                    "variance_adaptor_kernel_size": 3,
                    "pitch_bank_param_dict": {"kernel_size":3,"drop_out":0.5},
                    "energy_bank_param_dict": {"kernel_size":3,"drop_out":0.5}
                }
    kwargs_dict = {
                    "use_postnet": True
                  }
    model = Fastspeech2(*args_dict.values(), **kwargs_dict)
    print ("\n##############SequenceLe#############\n")
    print (sequence_lengths)

    ## Test inferencing, i.e no true duration, pitch, energy
    model.eval()
    mels_post, mels, duration_p, pitch_p, energy_p, encoder_alignments, decoder_alignments = model(batch, sequence_lengths, input_mask=mask)
    print ("\n##############VA Inference Tests#############\n")
    print (mels_post.shape, duration_p.shape, pitch_p.shape, energy_p.shape)

    # Test training, i.e given duration, pitch, energy
    model.train()
    durations = torch.randint(low=2, high=6, size=batch.size())
    duration_max = [duration[:idx].sum() for duration, idx in zip(durations, sequence_lengths)]
    durations_max = max(duration_max)

    pitch = torch.randn(size=(batch_size, durations_max)).unsqueeze(-1)
    energy = torch.randn(size=(batch_size, durations_max)).unsqueeze(-1)
    mels_post, mels, duration_p, pitch_p, energy_p, encoder_alignments, decoder_alignments = model(batch, sequence_lengths, 
                                                        label_durations=durations, label_pitch=pitch, label_energy=energy)

    print("\n##############VA Train Tests#############\n")
    print(mels_post.shape, duration_p.shape, pitch_p.shape, energy_p.shape)

def test_fastspeech2Loss(batch_size, do_mask):
    config = AttrDict({'loss_masking': True, "seq_len_norm": False})
    batch, sequence_lengths, mask = gen_batch_mask(batch_size)
    args_dict = {
                    "num_inputs": 40,
                    "num_out_mels": 80,
                    "num_input_channels": batch.size(-1),
                    "encoder_kernel_size": 3,
                    "decoder_kernel_size": 3,
                    "variance_adaptor_kernel_size": 3,
                    "pitch_bank_param_dict": {"kernel_size":3,"drop_out":0.5},
                    "energy_bank_param_dict": {"kernel_size":3,"drop_out":0.5}
                }
    kwargs_dict = {
                    "use_postnet": True
                  }
    model = Fastspeech2(*args_dict.values(), **kwargs_dict)
    print ("\n##############SequenceLe#############\n")
    print (sequence_lengths,mask.shape)

    # Test training, i.e given duration, pitch, energy
    model.train()
    durations = torch.randint(low=2, high=6, size=batch.size())
    mel_lengths = torch.tensor([duration[:idx].sum() for duration, idx in zip(durations, sequence_lengths)])
    durations_max = mel_lengths.max()
    pitch = torch.randn(size=(batch_size, durations_max)).unsqueeze(-1)
    energy = torch.randn(size=(batch_size, durations_max)).unsqueeze(-1)

    mels_post, mels, duration_p, pitch_p, energy_p, encoder_alignments, decoder_alignments = model(batch, sequence_lengths, 
                                                        label_durations=durations, label_pitch=pitch, label_energy=energy)
    criterion = Fastspeech2Loss(config)
    loss_dict = criterion(mels, mels.detach(), duration_p, durations.unsqueeze(2), pitch_p, pitch, energy_p, energy, sequence_lengths, mel_lengths, 
                          mels_post = mels_post)
    loss_dict = {key:value.item() for key,value in loss_dict.items()}
    print ("\n##############Loss dict is#############\n")
    print (loss_dict)


if __name__ == "__main__":
    batch_size, num_tokens, embedding_dim = 4, 10, 10
    batch = gen_batch(batch_size, num_tokens, embedding_dim)
    DO_MASK = True

    #test_MultiheadedAttention(batch,do_mask=do_mask)
    #test_Convd1d2Layers(batch)
    #test_FFT(batch,do_mask=do_mask)
    #test_Transformer(batch, do_mask=DO_MASK)
    #test_variance_adaptor(batch, do_mask=DO_MASK)
    #batch = torch.randint(low=0,high=20,size=(batch_size, num_tokens))
    #test_fastspeech2(batch_size, do_mask=DO_MASK)
    test_fastspeech2Loss(batch_size, do_mask=DO_MASK)