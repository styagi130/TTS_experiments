import argparse
import pathlib
import unittest
import sys
import tqdm
sys.path.append("./..")

from TTS.utils.io import load_config
from TTS.tts.datasets.preprocess import load_meta_data
from TTS.tts.datasets.TTSDataset import MyDataset
from TTS.utils.audio import AudioProcessor

class unitTestDataSet(unittest.TestCase):
    """
        Dataset Test class
    """
    def __init__(self,config_filepath:pathlib.Path, *args, **kwargs):
        super(unitTestDataSet, self).__init__(*args, **kwargs)

        self.config = load_config(str(config_filepath))
        self.dataset_train = self.__setup_my_dataset(self.config.r,train=True)
        self.dataset_eval = self.__setup_my_dataset(self.config.r,train=False)

    def __setup_my_dataset(self,r, verbose=False,train=True):
        meta_data_train, meta_data_eval = load_meta_data(self.config.datasets)
        ap = AudioProcessor(**self.config.audio)
        return MyDataset(
                        r,
                        self.config.text_cleaner,
                        compute_linear_spec=True if self.config.model.lower() == 'tacotron' else False,
                        meta_data=meta_data_train if train else meta_data_eval,
                        ap=ap,
                        tp=self.config.characters if 'characters' in self.config.keys() else None,
                        batch_group_size = self.config.batch_group_size * self.config.batch_size,
                        max_seq_len = self.config.max_seq_len,
                        min_seq_len = self.config.min_seq_len,
                        phoneme_cache_path=self.config.phoneme_cache_path,
                        use_phonemes=self.config.use_phonemes,
                        phoneme_language=self.config.phoneme_language,
                        enable_eos_bos=self.config.enable_eos_bos_chars,
                        use_mels_from_file=self.config.use_mels_from_file,
                        return_pitch=self.config.return_pitch,
                        return_energy=self.config.return_energy,
                        return_duration=self.config.return_duration,
                        mel_dir=self.config.mel_dir,
                        energy_dir=self.config.energy_dir,
                        pitch_dir=self.config.pitch_dir,
                        duration_dir=self.config.duration_dir,
                        verbose=verbose
                        )

    def _iter_tests(self, train = False):
        dataset = self.dataset_train if train else self.dataset_eval
        for i in range(len(dataset.items)):
            dataset_ele = dataset[i]
            assert "wav_stem" in dataset_ele.keys(), f"wave_stem did not generate for {dataset_ele['item_idx']}"
    
    """def test_train_iter(self):
        self._iter_tests(train = True)

    def test_eval_iter(self):
        self._iter_tests(train = False)"""
    
    def _batch_tests(self, train=False):
        dataset = self.dataset_train if train else self.dataset_eval
        
        # generate batches
        batches_range = [(idx, idx+self.config.batch_size) for idx in range(len(dataset.items)-self.config.batch_size)]
        for index, batch_range in tqdm.tqdm(enumerate(batches_range), total=len(batches_range), desc="train" if train else "eval", ncols=100):
            batch = []
            for i in range(batch_range[0], batch_range[1]):
                batch.append(dataset[i])

            text, text_lenghts, speaker_name, _, mel, mel_lengths, stop_targets, item_idxs, energy, pitch, duration = dataset.collate_fn(batch)

            assert text.size() == duration.size(), f"mismatch between tokens shape: {text.size()} and duration shape: {duration.size()}"
            assert mel.size()[:-1] == energy.size(), f"mismatch between mel shape: {mel.size()} and energy shape: {energy.size()}"
            assert mel.size()[:-1] == pitch.size(), f"mismatch between mel shape: {mel.size()} and pitch shape: {pitch.size()}"

            for i in range(mel.size(0)):
                assert duration[i].sum().item() == mel_lengths.data[i], f"Frame duration mismatch {duration[i].shape}, mel_size: {mel_lengths[i]} for batch index: {index} and sub index{i}"

    def test_batch_train(self):
        self._batch_tests(train = True)

    def test_batch_eval(self):
        self._batch_tests(train = False)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Dataloader unit tests")
    parser.add_argument("config_file",metavar="fc", type=pathlib.Path, help="Config filepath")

    args = parser.parse_args()
    config_filepath = args.config_file

    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(unitTestDataSet)
    
    test_suite = unittest.TestSuite()
    for test_name in test_names:
        test_suite.addTest(unitTestDataSet(config_filepath, test_name))

    unittest.TextTestRunner().run(test_suite)