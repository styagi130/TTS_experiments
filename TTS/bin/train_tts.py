#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append("../../")
from TTS.tts.datasets.preprocess import load_meta_data
from TTS.tts.datasets.TTSDataset import MyDataset
from TTS.tts.layers.losses import TacotronLoss, Fastspeech2Loss
from TTS.tts.utils.console_logger import ConsoleLogger
from TTS.tts.utils.distribute import (DistributedSampler,
                                      apply_gradient_allreduce,
                                      init_distributed, reduce_tensor)
from TTS.tts.utils.generic_utils import check_config, setup_model
from TTS.tts.utils.io import save_best_model, save_checkpoint
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.speakers import (get_speakers, load_speaker_mapping,
                                    save_speaker_mapping)
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import (KeepAverage, count_parameters,
                                     create_experiment_folder, get_git_branch,
                                     remove_experiment_folder, set_init_dict)
from TTS.utils.io import copy_config_file, load_config
from TTS.utils.radam import RAdam
from TTS.utils.tensorboard_logger import TensorboardLogger
from TTS.utils.training import (NoamLR, adam_weight_decay, check_update,
                                gradual_training_scheduler, set_weight_decay,
                                setup_torch_training_env)

use_cuda, num_gpus = setup_torch_training_env(True, False)


def setup_loader(ap, r, is_val=False, verbose=False):
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = MyDataset(
            r,
            c.text_cleaner,
            compute_linear_spec=True if c.model.lower() == 'tacotron' else False,
            meta_data=meta_data_eval if is_val else meta_data_train,
            ap=ap,
            tp=c.characters if 'characters' in c.keys() else None,
            batch_group_size=0 if is_val else c.batch_group_size *
            c.batch_size,
            min_seq_len=c.min_seq_len,
            max_seq_len=c.max_seq_len,
            phoneme_cache_path=c.phoneme_cache_path,
            use_phonemes=c.use_phonemes,
            phoneme_language=c.phoneme_language,
            enable_eos_bos=c.enable_eos_bos_chars,
            use_mels_from_file=c.use_mels_from_file,
            return_pitch=c.return_pitch,
            return_energy=c.return_energy,
            return_duration=c.return_duration,
            mel_dir=c.mel_dir,
            energy_dir=c.energy_dir_val if is_val else c.energy_dir,
            pitch_dir=c.pitch_dir_val if is_val else c.pitch_dir,
            duration_dir=c.duration_dir,
            verbose=verbose)
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=c.eval_batch_size if is_val else c.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            sampler=sampler,
            num_workers=c.num_val_loader_workers
            if is_val else c.num_loader_workers,
            pin_memory=False)
    return loader


def format_data(data):
    if c.use_speaker_embedding:
        speaker_mapping = load_speaker_mapping(OUT_PATH)

    # setup input data
    text_input = data[0]
    text_lengths = data[1]
    speaker_names = data[2]
    linear_input = data[3] if c.model in ["Tacotron"] else None
    mel_input = data[4]
    mel_lengths = data[5]
    stop_targets = data[6]
    energy = data[7]
    pitch = data[8]
    duration = data[9]
    avg_text_length = torch.mean(text_lengths.float())
    avg_spec_length = torch.mean(mel_lengths.float())

    if c.use_speaker_embedding:
        speaker_ids = [
            speaker_mapping[speaker_name] for speaker_name in speaker_names
        ]
        speaker_ids = torch.LongTensor(speaker_ids)
    else:
        speaker_ids = None

    # set stop targets view, we predict a single stop token per iteration.
    stop_targets = stop_targets.view(text_input.shape[0],
                                     stop_targets.size(1) // c.r, -1)
    stop_targets = (stop_targets.sum(2) >
                    0.0).unsqueeze(2).float().squeeze(2)

    # dispatch data to GPU
    if use_cuda:
        text_input = text_input.cuda(non_blocking=True)
        text_lengths = text_lengths.cuda(non_blocking=True)
        mel_input = mel_input.cuda(non_blocking=True)
        mel_lengths = mel_lengths.cuda(non_blocking=True)
        linear_input = linear_input.cuda(non_blocking=True) if c.model in ["Tacotron"] else None
        stop_targets = stop_targets.cuda(non_blocking=True)
        energy = energy.cuda(non_blocking=True) if c.model.lower() in ["fastspeech2"] else None
        pitch = pitch.cuda(non_blocking=True) if c.model.lower() in ["fastspeech2"] else None
        duration = duration.cuda(non_blocking=True) if c.model.lower() in ["fastspeech2"] else None
        if speaker_ids is not None:
            speaker_ids = speaker_ids.cuda(non_blocking=True)
    return text_input, text_lengths, mel_input, mel_lengths, linear_input, stop_targets, energy, pitch, duration, speaker_ids, avg_text_length, avg_spec_length



def train(model, criterion, optimizer, optimizer_st, scheduler,
          c, ap, global_step, epoch):
    data_loader = setup_loader(ap, model.decoder.r, is_val=False,
                               verbose=(epoch == 0))
    model.train()
    epoch_time = 0
    keep_avg = KeepAverage()
    if use_cuda:
        batch_n_iter = int(
            len(data_loader.dataset) / (c.batch_size * num_gpus))
    else:
        batch_n_iter = int(len(data_loader.dataset) / c.batch_size)
    end_time = time.time()
    c_logger.print_train_start()
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # format data
        text_input, text_lengths, mel_input, mel_lengths, linear_input, stop_targets, energy, pitch, duration, speaker_ids, avg_text_length, avg_spec_length = format_data(data)

        loader_time = time.time() - end_time

        global_step += 1

        # setup lr
        if c.noam_schedule:
            scheduler.step()
        optimizer.zero_grad()
        if optimizer_st:
            optimizer_st.zero_grad()

        # forward pass model
        if c.model.lower() == "fastspeech2":
            _, mels_post, duration_p, pitch_p, energy_p, _, _ = model(text_input, text_lengths, label_durations=duration, label_pitch=pitch,  label_energy=energy)
        else:
            if c.bidirectional_decoder or c.double_decoder_consistency:
                decoder_output, postnet_output, alignments, stop_tokens, decoder_backward_output, alignments_backward = model(
                    text_input, text_lengths, mel_input, mel_lengths, speaker_ids=speaker_ids)
            else:
                decoder_output, postnet_output, alignments, stop_tokens = model(
                    text_input, text_lengths, mel_input, mel_lengths, speaker_ids=speaker_ids)
                decoder_backward_output = None
                alignments_backward = None

        # set the alignment lengths wrt reduction factor for guided attention
        if "tacotron" in c.model.lower():
            if mel_lengths.max() % model.decoder.r != 0:
                alignment_lengths = (mel_lengths + (model.decoder.r - (mel_lengths.max() % model.decoder.r))) // model.decoder.r
            else:
                alignment_lengths = mel_lengths //  model.decoder.r

        # compute loss
        if c.model.lower() == "fastspeech2":
            loss_dict = criterion(mels_post, mel_input, duration_p, duration, pitch_p, pitch, energy_p, energy, text_lengths, mel_lengths)
        else:
            loss_dict = criterion(postnet_output, decoder_output, mel_input,
                              linear_input, stop_tokens, stop_targets,
                              mel_lengths, decoder_backward_output,
                              alignments, alignment_lengths, alignments_backward,
                              text_lengths)

        # backward pass
        loss_dict['loss'].backward()
        optimizer, current_lr = adam_weight_decay(optimizer)
        grad_norm, _ = check_update(model, c.grad_clip, ignore_stopnet=True)
        optimizer.step()

        # compute alignment error (the lower the better )
        if "tacotron" in c.model.lower():
            align_error = 1 - alignment_diagonal_score(alignments)
            loss_dict['align_error'] = align_error

            # backpass and check the grad norm for stop loss
            if c.separate_stopnet:
                loss_dict['stopnet_loss'].backward()
                optimizer_st, _ = adam_weight_decay(optimizer_st)
                grad_norm_st, _ = check_update(model.decoder.stopnet, 1.0)
                optimizer_st.step()
            else:
                grad_norm_st = 0

        step_time = time.time() - start_time
        epoch_time += step_time

        # aggregate losses from processes
        if num_gpus > 1:
            if "tacotron" in c.model.lower():
                loss_dict['postnet_loss'] = reduce_tensor(loss_dict['postnet_loss'].data, num_gpus)
                loss_dict['decoder_loss'] = reduce_tensor(loss_dict['decoder_loss'].data, num_gpus)
                loss_dict['loss'] = reduce_tensor(loss_dict['loss'] .data, num_gpus)
                loss_dict['stopnet_loss'] = reduce_tensor(loss_dict['stopnet_loss'].data, num_gpus) if c.stopnet else loss_dict['stopnet_loss']
            if "fastspeech2" == c.model.lower():
                loss_dict['mel_loss'] = reduce_tensor(loss_dict['postnet_loss'].data, num_gpus)
                loss_dict['va_loss'] = reduce_tensor(loss_dict['va_loss'].data, num_gpus)
                loss_dict['loss'] = reduce_tensor(loss_dict['loss'] .data, num_gpus)
                loss_dict['duration_loss'] = reduce_tensor(loss_dict['duration_loss'].data, num_gpus)
                loss_dict['pitch_loss'] = reduce_tensor(loss_dict['pitch_loss'].data, num_gpus)
                loss_dict['energy_loss'] = reduce_tensor(loss_dict['energy_loss'].data, num_gpus)


        # detach loss values
        loss_dict_new = dict()
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                loss_dict_new[key] = value
            else:
                loss_dict_new[key] = value.item()
        loss_dict = loss_dict_new

        # update avg stats
        update_train_values = dict()
        for key, value in loss_dict.items():
            update_train_values['avg_' + key] = value
        update_train_values['avg_loader_time'] = loader_time
        update_train_values['avg_step_time'] = step_time
        keep_avg.update_values(update_train_values)

        # print training progress
        if global_step % c.print_step == 0:
            c_logger.print_train_step(batch_n_iter, num_iter, global_step,
                                      avg_spec_length, avg_text_length,
                                      step_time, loader_time, current_lr,
                                      loss_dict, keep_avg.avg_values)

        if args.rank == 0:
            # Plot Training Iter Stats
            # reduce TB load
            if global_step % c.tb_plot_step == 0:
                if "tacotron" in c.model.lower():
                    iter_stats = {
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                        "grad_norm_st": grad_norm_st,
                        "step_time": step_time
                    }
                elif "fastspeech2" in c.model.lower():
                    iter_stats = {
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                        "step_time": step_time
                    }
                iter_stats.update(loss_dict)
                tb_logger.tb_train_iter_stats(global_step, iter_stats)

            if global_step % c.save_step == 0:
                if c.checkpoint:
                    # save model
                    if "tacotron" in c.model.lower():
                        save_checkpoint(model, optimizer, global_step, epoch, model.decoder.r, OUT_PATH,
                                        optimizer_st=optimizer_st,
                                        model_loss=loss_dict['postnet_loss'])
                    elif "fastspeech2" in c.model.lower():
                        save_checkpoint(model, optimizer, global_step, epoch, model.decoder.r, OUT_PATH,
                                        optimizer_st=optimizer_st,
                                        model_loss=loss_dict['mel_loss'])

                # Diagnostic visualizations
                const_spec = mels_post[0].data.cpu().numpy() if "fastspeech2" in c.model.lower() else postnet_output[0].data.cpu().numpy()
                gt_spec = linear_input[0].data.cpu().numpy() if c.model in [
                    "Tacotron", "TacotronGST"
                ] else mel_input[0].data.cpu().numpy()
                if "tacotron" in c.model.lower():
                    align_img = alignments[0].data.cpu().numpy()

                    figures = {
                        "prediction": plot_spectrogram(const_spec, ap),
                        "ground_truth": plot_spectrogram(gt_spec, ap),
                        "alignment": plot_alignment(align_img),
                    }
                    if c.bidirectional_decoder or c.double_decoder_consistency:
                        figures["alignment_backward"] = plot_alignment(alignments_backward[0].data.cpu().numpy())
                elif "fastspeech2" in c.model.lower():
                    figures = {
                        "prediction": plot_spectrogram(const_spec, ap),
                        "ground_truth": plot_spectrogram(gt_spec, ap),
                    }

                tb_logger.tb_train_figures(global_step, figures)

                # Sample audio
                if c.model in ["Tacotron", "TacotronGST"]:
                    train_audio = ap.inv_spectrogram(const_spec.T)
                else:
                    train_audio = ap.inv_melspectrogram(const_spec.T)
                tb_logger.tb_train_audios(global_step,
                                          {'TrainAudio': train_audio},
                                          c.audio["sample_rate"])
        end_time = time.time()

    # print epoch stats
    c_logger.print_train_epoch_end(global_step, epoch, epoch_time, keep_avg)

    # Plot Epoch Stats
    if args.rank == 0:
        epoch_stats = {"epoch_time": epoch_time}
        epoch_stats.update(keep_avg.avg_values)
        tb_logger.tb_train_epoch_stats(global_step, epoch_stats)
        if c.tb_model_param_stats:
            tb_logger.tb_model_weights(model, global_step)
    return keep_avg.avg_values, global_step


@torch.no_grad()
def evaluate(model, criterion, c, ap, global_step, epoch):
    data_loader = setup_loader(ap, model.decoder.r, is_val=True)
    model.eval()
    epoch_time = 0
    keep_avg = KeepAverage()
    c_logger.print_eval_start()
    if data_loader is not None:
        for num_iter, data in enumerate(data_loader):
            start_time = time.time()

            # format data
            text_input, text_lengths, mel_input, mel_lengths, linear_input, stop_targets, energy, pitch, duration, speaker_ids, avg_text_length, avg_spec_length = format_data(data)
            if "tacotron" in c.model.lower():
                assert mel_input.shape[1] % model.decoder.r == 0

            # forward pass model
                if c.bidirectional_decoder or c.double_decoder_consistency:
                    decoder_output, postnet_output, alignments, stop_tokens, decoder_backward_output, alignments_backward = model(
                        text_input, text_lengths, mel_input, speaker_ids=speaker_ids)
                else:
                    decoder_output, postnet_output, alignments, stop_tokens = model(
                        text_input, text_lengths, mel_input, speaker_ids=speaker_ids)
                    decoder_backward_output = None
                    alignments_backward = None

                # set the alignment lengths wrt reduction factor for guided attention
                if mel_lengths.max() % model.decoder.r != 0:
                    alignment_lengths = (mel_lengths + (model.decoder.r - (mel_lengths.max() % model.decoder.r))) // model.decoder.r
                else:
                    alignment_lengths = mel_lengths //  model.decoder.r

                # compute loss
                loss_dict = criterion(postnet_output, decoder_output, mel_input,
                                    linear_input, stop_tokens, stop_targets,
                                    mel_lengths, decoder_backward_output,
                                    alignments, alignment_lengths, alignments_backward,
                                    text_lengths)
            elif c.model.lower() == "fastspeech2":
                _, mels_post, duration_p, pitch_p, energy_p, _, _ = model(text_input, text_lengths, label_durations=duration, label_pitch=pitch,  label_energy=energy)
                loss_dict = criterion(mels_post, mel_input, duration_p, duration, pitch_p, pitch, energy_p, energy, text_lengths, mel_lengths)



            # step time
            step_time = time.time() - start_time
            epoch_time += step_time

            # compute alignment score
            if "tacotron" in c.model.lower():
                align_error = 1 - alignment_diagonal_score(alignments)
                loss_dict['align_error'] = align_error

                # aggregate losses from processes
                if num_gpus > 1:
                    loss_dict['postnet_loss'] = reduce_tensor(loss_dict['postnet_loss'].data, num_gpus)
                    loss_dict['decoder_loss'] = reduce_tensor(loss_dict['decoder_loss'].data, num_gpus)
                    if c.stopnet:
                        loss_dict['stopnet_loss'] = reduce_tensor(loss_dict['stopnet_loss'].data, num_gpus)
            elif "fastspeech2" == c.model.lower() and num_gpus > 1:
                loss_dict['mel_loss'] = reduce_tensor(loss_dict['postnet_loss'].data, num_gpus)
                loss_dict['va_loss'] = reduce_tensor(loss_dict['va_loss'].data, num_gpus)
                loss_dict['loss'] = reduce_tensor(loss_dict['loss'] .data, num_gpus)
                loss_dict['duration_loss'] = reduce_tensor(loss_dict['duration_loss'].data, num_gpus)
                loss_dict['pitch_loss'] = reduce_tensor(loss_dict['pitch_loss'].data, num_gpus)
                loss_dict['energy_loss'] = reduce_tensor(loss_dict['energy_loss'].data, num_gpus)

            # detach loss values
            loss_dict_new = dict()
            for key, value in loss_dict.items():
                if isinstance(value, (int, float)):
                    loss_dict_new[key] = value
                else:
                    loss_dict_new[key] = value.item()
            loss_dict = loss_dict_new

            # update avg stats
            update_train_values = dict()
            for key, value in loss_dict.items():
                update_train_values['avg_' + key] = value
            keep_avg.update_values(update_train_values)

            if c.print_eval:
                c_logger.print_eval_step(num_iter, loss_dict, keep_avg.avg_values)

        if args.rank == 0:
            # Diagnostic visualizations
            idx = np.random.randint(mel_input.shape[0])
            const_spec = mels_post[idx].data.cpu().numpy() if "fastspeech2" in c.model.lower() else postnet_output[idx].data.cpu().numpy() 
            gt_spec = linear_input[idx].data.cpu().numpy() if c.model in [
                "Tacotron", "TacotronGST"
            ] else mel_input[idx].data.cpu().numpy()

            if "tacotron" in c.model.lower():
                align_img = alignments[idx].data.cpu().numpy()
                eval_figures = {
                    "prediction": plot_spectrogram(const_spec, ap),
                    "ground_truth": plot_spectrogram(gt_spec, ap),
                    "alignment": plot_alignment(align_img)
                }
            elif "fastspeech2" in c.model.lower():
                eval_figures = {
                    "prediction": plot_spectrogram(const_spec, ap),
                    "ground_truth": plot_spectrogram(gt_spec, ap),
                }

            # Sample audio
            if c.model in ["Tacotron", "TacotronGST"]:
                eval_audio = ap.inv_spectrogram(const_spec.T)
            else:
                eval_audio = ap.inv_melspectrogram(const_spec.T)
            tb_logger.tb_eval_audios(global_step, {"ValAudio": eval_audio},
                                     c.audio["sample_rate"])

            # Plot Validation Stats
            if "tacotron" in c.model.lower():
                if c.bidirectional_decoder or c.double_decoder_consistency:
                    align_b_img = alignments_backward[idx].data.cpu().numpy()
                    eval_figures['alignment2'] = plot_alignment(align_b_img)
            tb_logger.tb_eval_stats(global_step, keep_avg.avg_values)
            tb_logger.tb_eval_figures(global_step, eval_figures)

    if args.rank == 0 and epoch > c.test_delay_epochs:
        if c.test_sentences_file is None:
            test_sentences = [
				"सॉलोमॉन्ज़ वॉज़ नॉट स्ट्रेट्फॉर्वर्ड इन हिज़ रिप्लाइज़ , ऐज़ टु वेअर ही गॉट द गोल्ड , ऐंड ही वॉज़ सून प्लेस्ट इन द डॉक , विद द कैस्पर्ज़ ऐंड मॉस .",
				"वॉट्सैप नम्बर है मेरे पास , सेव किया हूँ , उसको बात हुए थी उसपे .",
				"हाँ , हम प्रोसेस करूंगा , अगर कुछ जरुरत हमें फोन करना , लेकिन अभी से बात करना , हमें वो नेहा , स्नेहा किसी को मत देना , अगर नेहा भी पकड़ो कोई प्रिया भी पकड़ो , मिस शिवानी को ही देने का , ठीक हैं ना .",
				"गुड आफ्टर्नून, माइ नेम इज़ अलीशा यॉर कॉल हैज़ लैंडेड टु इंडिआमार्ट बाएअर हेल्प्डेस्क, हाउ मे आइ असिस्ट यू? बताइये क्या खरीदना चाहते हैं?",
				"तेंदुलकर ने साथ ही कहा , की मौजूद भारतीय टीम इकाई के रूप में कहीं अधिक संतुलित है.",
				"येस इग्ज़ैक्ट्ली, ऐंड देन आफ्टर दैट , वी विल रिफंड इट बट इट विल टेक अटलीस्ट फॉर्टी एट आउअर्ज़ टु अपडेट द करेंट रिफंड अमाउंट.",
				"माइ स्पेशल्टी इज़ इन वॉट आइ कॉल , सोल प्रॉस्पेक्टर्ज़, ए क्रॉस - ऐक्सिअल क्लैसिफिकेशन , आइ हैव कोडिफाइड बाइ एक्स्टेंसिव इंटरैक्शन विद विज़िटर्ज़, लाइक यॉर्सेल्फ़.",
				"कॉन्सेक्रेटेड टु द संस्कृत महारिशी पनीनी, ऑल्सो नॉट ऐन ऑर्डिन्री लिंग्विस्ट, आर्किटेक्ट ऑव महारिशी, महारिशी कट्यान, ऐंड योगा साइंस इज़, पनहिजिल."
            ]
        else:
            with open(c.test_sentences_file, "r") as f:
                test_sentences = [s.strip() for s in f.readlines()]

        # test sentences
        test_audios = {}
        test_figures = {}
        print(" | > Synthesizing test sentences")
        speaker_id = 0 if c.use_speaker_embedding else None
        style_wav = c.get("style_wav_for_test")
        for idx, test_sentence in enumerate(test_sentences):
            try:
                wav, alignment, decoder_output, postnet_output, stop_tokens, inputs = synthesis(
                    model,
                    test_sentence,
                    c,
                    use_cuda,
                    ap,
                    speaker_id=speaker_id,
                    style_wav=style_wav,
                    truncated=False,
                    enable_eos_bos_chars=c.enable_eos_bos_chars, #pylint: disable=unused-argument
                    use_griffin_lim=True,
                    do_trim_silence=False)

                file_path = os.path.join(AUDIO_PATH, str(global_step))
                os.makedirs(file_path, exist_ok=True)
                file_path = os.path.join(file_path,
                                         "TestSentence_{}.wav".format(idx))
                ap.save_wav(wav, file_path)
                test_audios['{}-audio'.format(idx)] = wav
                test_figures['{}-prediction'.format(idx)] = plot_spectrogram(
                    postnet_output, ap)
                if "tacotron" in c.model.lower():
                    test_figures['{}-alignment'.format(idx)] = plot_alignment(
                        alignment)
            except:
                print(" !! Error creating Test Sentence -", idx)
                traceback.print_exc()
        tb_logger.tb_test_audios(global_step, test_audios,
                                 c.audio['sample_rate'])
        tb_logger.tb_test_figures(global_step, test_figures)
    return keep_avg.avg_values


# FIXME: move args definition/parsing inside of main?
def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global meta_data_train, meta_data_eval, symbols, phonemes
    # Audio processor
    ap = AudioProcessor(**c.audio)
    if 'characters' in c.keys():
        symbols, phonemes = make_symbols(**c.characters)

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])
    num_chars = len(phonemes) if c.use_phonemes else len(symbols)

    # load data instances
    meta_data_train, meta_data_eval = load_meta_data(c.datasets)

    # set the portion of the data used for training
    if 'train_portion' in c.keys():
        meta_data_train = meta_data_train[:int(len(meta_data_train) * c.train_portion)]
    if 'eval_portion' in c.keys():
        meta_data_eval = meta_data_eval[:int(len(meta_data_eval) * c.eval_portion)]

    # parse speakers
    if c.use_speaker_embedding:
        speakers = get_speakers(meta_data_train)
        if args.restore_path:
            prev_out_path = os.path.dirname(args.restore_path)
            speaker_mapping = load_speaker_mapping(prev_out_path)
            assert all([speaker in speaker_mapping
                        for speaker in speakers]), "As of now you, you cannot " \
                                                   "introduce new speakers to " \
                                                   "a previously trained model."
        else:
            speaker_mapping = {name: i for i, name in enumerate(speakers)}
        save_speaker_mapping(OUT_PATH, speaker_mapping)
        num_speakers = len(speaker_mapping)
        print("Training with {} speakers: {}".format(num_speakers,
                                                     ", ".join(speakers)))
    else:
        num_speakers = 0

    model = setup_model(num_chars, num_speakers, c)

    params = set_weight_decay(model, c.wd)
    optimizer = RAdam(params, lr=c.lr, weight_decay=0)
    optimizer_st = None
    if "tacotron" in c.model.lower():
        if c.stopnet and c.separate_stopnet:
            optimizer_st = RAdam(model.decoder.stopnet.parameters(),
                                lr=c.lr,
                                weight_decay=0)
    # setup criterion
    if "tacotron" in c.model.lower():
        criterion = TacotronLoss(c, stopnet_pos_weight=10.0, ga_sigma=0.4)
    elif "fastspeech2" == c.model.lower():
        criterion = Fastspeech2Loss(c)

    if args.restore_path:
        checkpoint = torch.load(args.restore_path, map_location='cpu')
        try:
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if c.reinit_layers:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
        except:
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint['model'], c)
            model.load_state_dict(model_dict)
            del model_dict
        for group in optimizer.param_groups:
            group['lr'] = c.lr
        print(" > Model restored from step %d" % checkpoint['step'],
              flush=True)
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0

    if use_cuda:
        model.cuda()
        criterion.cuda()

    # DISTRUBUTED
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    if c.noam_schedule:
        scheduler = NoamLR(optimizer,
                           warmup_steps=c.warmup_steps,
                           last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    global_step = args.restore_step
    for epoch in range(0, c.epochs):
        c_logger.print_epoch_start(epoch, c.epochs)
        # set gradual training
        if c.gradual_training is not None:
            r, c.batch_size = gradual_training_scheduler(global_step, c)
            c.r = r
            model.decoder.set_r(r)
            if c.bidirectional_decoder:
                model.decoder_backward.set_r(r)
            print("\n > Number of output frames:", model.decoder.r)
        train_avg_loss_dict, global_step = train(model, criterion, optimizer,
                                                 optimizer_st, scheduler, c, ap,
                                                 global_step, epoch)
        eval_avg_loss_dict = evaluate(model, criterion, c, ap, global_step, epoch)
        c_logger.print_epoch_end(epoch, eval_avg_loss_dict)
        target_loss = train_avg_loss_dict['avg_mel_loss'] if "fastspeech2" == c.model.lower() else train_avg_loss_dict['avg_postnet_loss']
        if c.run_eval:
            target_loss = eval_avg_loss_dict['avg_mel_loss'] if "fastspeech2" == c.model.lower() else eval_avg_loss_dict['avg_postnet_loss']
        best_loss = save_best_model(target_loss, best_loss, model, optimizer, global_step, epoch, c.r,
                                    OUT_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--continue_path',
        type=str,
        help='Training output folder to continue training. Use to continue a training. If it is used, "config_path" is ignored.',
        default='',
        required='--config_path' not in sys.argv)
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Model file to be restored. Use to finetune a model.',
        default='')
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
        required='--continue_path' not in sys.argv
    )
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        help='Do not verify commit integrity to run training.')

    # DISTRUBUTED
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help='DISTRIBUTED: process rank for distributed training.')
    parser.add_argument('--group_id',
                        type=str,
                        default="",
                        help='DISTRIBUTED: process group id.')
    args = parser.parse_args()

    if args.continue_path != '':
        args.output_path = args.continue_path
        args.config_path = os.path.join(args.continue_path, 'config.json')
        list_of_files = glob.glob(args.continue_path + "/*.pth.tar") # * means all if need specific format then *.csv
        latest_model_file = max(list_of_files, key=os.path.getctime)
        args.restore_path = latest_model_file
        print(f" > Training continues for {args.restore_path}")

    # setup output paths and read configs
    c = load_config(args.config_path)
    check_config(c)
    _ = os.path.dirname(os.path.realpath(__file__))

    OUT_PATH = args.continue_path
    if args.continue_path == '':
        OUT_PATH = create_experiment_folder(c.output_path, c.run_name, args.debug)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

    c_logger = ConsoleLogger()

    if args.rank == 0:
        os.makedirs(AUDIO_PATH, exist_ok=True)
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        copy_config_file(args.config_path,
                         os.path.join(OUT_PATH, 'config.json'), new_fields)
        os.chmod(AUDIO_PATH, 0o775)
        os.chmod(OUT_PATH, 0o775)

        LOG_DIR = OUT_PATH
        tb_logger = TensorboardLogger(LOG_DIR, model_name='TTS')

        # write model desc to tensorboard
        tb_logger.tb_add_text('model-description', c['run_description'], 0)

    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
