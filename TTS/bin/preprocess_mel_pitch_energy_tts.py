#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
import yaml
import shutil
import argparse
import matplotlib.pyplot as plt
import math, pickle, os, glob
import numpy as np
import pathlib
from tqdm import tqdm
import sys
sys.path.append("./../../")
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"

def get_files(path, extension=".wav"):
    filenames = []
    for filename in glob.iglob(f"{path}/**/*{extension}", recursive=True):
        filenames += [filename]
    return filenames


def _process_file(path):
    wav = ap.load_wav(path)
    mel, energy, f0 = ap.get_mel_energy_pitch(wav)
    # check
    assert len(wav.shape) == 1, \
        f"{path} seems to be multi-channel signal."
    assert np.abs(wav).max() <= 1.0, \
        f"{path} seems to be different from 16 bit PCM."
    
    assert energy.shape[0] == mel.shape[1],f"num of Mel Frames: {mel.shape[1]} does not equal num energy frames: {energy.shape[0]} for file {path}"
    assert f0.shape[0] == mel.shape[1],f"num of Mel Frames: {mel.shape[1]} does not equal num pitch frames: {f0.shape[0]} for file {path}"

    # gap when wav is not multiple of hop_length
    gap = wav.shape[0] % ap.hop_length
    assert mel.shape[1] * ap.hop_length == wav.shape[0] + ap.hop_length - gap, f'{mel.shape[1] * ap.hop_length} vs {wav.shape[0] + ap.hop_length + gap}'
    return mel.astype(np.float32), energy.astype(np.float32), f0.astype(np.float32), wav


def extract_feats(wav_path, mel_path, energy_path, pitch_path):
    m, energy, f0, wav = _process_file(wav_path)

    np.save(mel_path, m, allow_pickle=False)
    np.save(energy_path, energy, allow_pickle=False)
    np.save(pitch_path, f0, allow_pickle=False)

    return wav_path, mel_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="path to config file for feature extraction."
    )
    parser.add_argument(
        "--num_procs", type=int, default=4, help="number of parallel processes."
    )
    args = parser.parse_args()

    # load config
    config = load_config(args.config_path)
    config.update(vars(args))

    config.audio['do_trim_silence'] = False
    # config['audio']['signal_norm'] = False  # do not apply earlier normalization

    ap = AudioProcessor(**config['audio'])

    for dataset in config['datasets']:
        basepath = pathlib.Path(dataset["basepath"])
        MEL_PATH = basepath / dataset["mel_dir"]
        Energy_Path = basepath / dataset["energy_dir"]
        Pitch_Path = basepath / dataset["pitch_dir"]
        # TODO: use TTS data processors
        seg_file_path = os.path.join(dataset["basepath"], dataset["audio_dir"])
        wav_files = get_files(seg_file_path)
        print(" > Number of audio files : {}".format(len(wav_files)))

        wav_file = wav_files[0]
        m, energy, f0, wav = _process_file(wav_file)

        # sanity check
        print(' > Sample Spec Stats...')
        print(' | > spectrogram max:', m.max())
        print(' | > spectrogram min: ', m.min())
        print(' | > spectrogram shape:', m.shape)
        print(' | > energy shape:', energy.shape, energy.dtype)
        print(' | > pitch shape:', f0.shape, f0.dtype)
        print(' | > wav shape:', wav.shape)
        print(' | > wav max - min:', wav.max(), ' - ', wav.min())

        # This will take a while depending on size of dataset
        #with Pool(args.num_procs) as p:
        #    dataset_ids = list(tqdm(p.imap(extract_feats, wav_files), total=len(wav_files)))
        dataset_ids = []
        for wav_file in tqdm(wav_files):
            file_stem = pathlib.Path(wav_file).stem
            
            mel_file = MEL_PATH / f"{file_stem}.npy"
            energy_file = Energy_Path / f"{file_stem}.npy"
            pitch_file = Pitch_Path / f"{file_stem}.npy"
            extract_feats(wav_file, mel_file, energy_file, pitch_file)
        # save metadata
        #with open(os.path.join(OUT_PATH, "metadata.txt"), "w") as f:
        #   for data in dataset_ids:
        #      f.write(f"{data[0]}|{data[1]}\n")
