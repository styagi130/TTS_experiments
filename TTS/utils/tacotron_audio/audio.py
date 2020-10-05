import librosa
import soundfile as sf
import numpy as np
import scipy.io
import scipy.signal
import pyworld as pw
from scipy.io.wavfile import read
import torch
from TTS.utils.tacotron_audio.torch_stft import TacotronSTFT

class AudioProcessor(object):
    def __init__(self,
                 sample_rate=None,
                 num_mels=None,
                 min_level_db=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 hop_length=None,
                 win_length=None,
                 ref_level_db=None,
                 fft_size=1024,
                 power=None,
                 preemphasis=0.0,
                 signal_norm=None,
                 symmetric_norm=None,
                 max_norm=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 spec_gain=20,
                 stft_pad_mode='reflect',
                 clip_norm=True,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 trim_db=60,
                 do_sound_norm=False,
                 stats_path=None,
                 **_):

        print(" > Setting up Audio Processor...")
        # setup class attributed
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db or 0
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.fft_size = fft_size
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = mel_fmin or 0
        self.mel_fmax = mel_fmax
        self.spec_gain = float(spec_gain)
        self.stft_pad_mode = 'reflect'
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db
        self.do_sound_norm = do_sound_norm
        self.stats_path = stats_path
        self.max_wav_value=32768.0
        # setup stft parameters
        if hop_length is None:
            # compute stft parameters from given time values
            self.hop_length, self.win_length = self._stft_parameters()
        else:
            # use stft parameters from config file
            self.hop_length = hop_length
            self.win_length = win_length
        assert min_level_db != 0.0, " [!] min_level_db is 0"
        assert self.win_length <= self.fft_size, " [!] win_length cannot be larger than fft_size"
        members = vars(self)
        for key, value in members.items():
            print(" | > {}:{}".format(key, value))
        # create spectrogram utils

        ## Define stft
        self.stft = TacotronSTFT(
                        self.fft_size, self.hop_length, self.win_length,
                        self.num_mels, self.sample_rate, self.mel_fmin,
                        self.mel_fmax)


    def _stft_parameters(self, ):
        """Compute necessary stft parameters with given time values"""
        factor = self.frame_length_ms / self.frame_shift_ms
        assert (factor).is_integer(), " [!] frame_shift_ms should divide frame_length_ms"
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(hop_length * factor)
        return hop_length, win_length

    def get_mel_energy_pitch(self,x):
        # calculate stft
        tor_x = torch.FloatTensor(x)
        tor_x = tor_x.unsqueeze(0)
        tor_x = torch.autograd.Variable(tor_x, requires_grad=False)
        mels = self.stft.mel_spectrogram(tor_x).detach().squeeze(0).cpu().numpy().astype(np.float32)

        energy = np.linalg.norm(np.abs(mels),axis=0)
        f0, t = pw.dio(x.astype(np.float64), self.sample_rate, frame_period = self.hop_length/self.sample_rate*1000)
        f0 = self.__adjust_f0_length(f0, mels)
        return mels, energy, f0

    def __adjust_f0_length(self, f0, mels):
        if f0.shape[0] < mels.shape[1]:
            return np.pad(f0, (0,mels.shape[1]-f0.shape[0]), "constant", constant_values=(0,0))
        return f0[:mels.shape[1]]


    def load_wav(self, filename):
        sampling_rate, data = read(filename)
        if sampling_rate != self.sample_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                   sampling_rate, self.sample_rate))
        data = data.astype(np.float32)
        data = data/ self.max_wav_value
        return data, sampling_rate
