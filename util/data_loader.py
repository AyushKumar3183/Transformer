"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import os
import shutil
import tarfile

import requests
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

MULTI30K_ROOT = '.data'
MULTI30K_DIR = os.path.join(MULTI30K_ROOT, 'multi30k')
MULTI30K_URLS = {
    'training.tar.gz': 'https://raw.githubusercontent.com/tanjeffreyz/pytorch-multi30k/main/training.tar.gz',
    'validation.tar.gz': 'https://raw.githubusercontent.com/tanjeffreyz/pytorch-multi30k/main/validation.tar.gz',
    'mmt16_task1_test.tar.gz': 'https://raw.githubusercontent.com/tanjeffreyz/pytorch-multi30k/main/mmt16_task1_test.tar.gz',
}
REQUIRED_FILES = (
    'train.en', 'train.de',
    'val.en', 'val.de',
    'test2016.en', 'test2016.de',
)


def _has_multi30k_files(data_dir):
    return all(os.path.isfile(os.path.join(data_dir, name)) for name in REQUIRED_FILES)


def _download_multi30k():
    os.makedirs(MULTI30K_DIR, exist_ok=True)

    for archive_name, url in MULTI30K_URLS.items():
        archive_path = os.path.join(MULTI30K_DIR, archive_name)
        if not os.path.isfile(archive_path):
            print(f'downloading {archive_name}')
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(archive_path, 'wb') as archive_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        archive_file.write(chunk)

        with tarfile.open(archive_path, 'r:gz') as archive:
            archive.extractall(path=MULTI30K_DIR)

    test_en = os.path.join(MULTI30K_DIR, 'test.en')
    test_de = os.path.join(MULTI30K_DIR, 'test.de')
    test2016_en = os.path.join(MULTI30K_DIR, 'test2016.en')
    test2016_de = os.path.join(MULTI30K_DIR, 'test2016.de')

    if os.path.isfile(test_en) and not os.path.exists(test2016_en):
        os.symlink('test.en', test2016_en)
    if os.path.isfile(test_de) and not os.path.exists(test2016_de):
        os.symlink('test.de', test2016_de)


def ensure_multi30k():
    if _has_multi30k_files(MULTI30K_DIR):
        return MULTI30K_DIR

    if os.path.isdir(MULTI30K_DIR):
        shutil.rmtree(MULTI30K_DIR)

    _download_multi30k()
    return MULTI30K_DIR


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        dataset_path = ensure_multi30k()
        train_data, valid_data, test_data = Multi30k.splits(
            exts=self.ext, fields=(self.source, self.target), path=dataset_path)
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
