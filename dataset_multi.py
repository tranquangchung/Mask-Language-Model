import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils import pad_1D, pad_2D
import pdb
import random
import copy
import torch
from utils import get_mask_from_lengths


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False, random=True
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.language, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        
        with open(os.path.join(self.preprocessed_path, "languages.json")) as f:
            self.language_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last
        self.random = random

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        lang = self.language[idx]
        lang_id = self.language_map[lang]
        raw_text = self.raw_text[idx]
        phone = text_to_sequence(self.text[idx], self.cleaners)
        input_model = copy.copy(phone)
        output_label = []
        for i, token in enumerate(input_model):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    input_model[i] = 4 # mask_token
                elif prob < 0.9 and self.random:
                    input_model[i] = random.randint(5, 1050) # tính cả 5 và 1050
                output_label.append(phone[i])
            else:
                output_label.append(0)
        phone = [3] + phone + [2]
        input_model = [3] + input_model + [2] # start and end token
        output_label = [0] + output_label + [0] # padding token
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{0}-{1}-mel-{2}.npy".format(lang, speaker, basename),
        )

        mel = np.load(mel_path)

        return input_model, output_label, mel

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            language = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, l, s, t, r = line.strip("\n").split("|")
                name.append(n)
                language.append(l)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, language, speaker, text, raw_text

    def collate_fn(self, batch):
        input_lens = [len(x[0]) for x in batch]
        max_x_len = max(input_lens)

        # chars
        chars_pad = [pad1d(x[0], max_x_len) for x in batch]
        chars = np.stack(chars_pad)

        # labels
        labels_pad = [pad1d(x[1], max_x_len) for x in batch]
        labels = np.stack(labels_pad)

        # position
        position = [pad1d(range(1, len + 1), max_x_len) for len in input_lens]
        position = np.stack(position)
        text_lens = np.array([len(x[0]) for x in batch])

        # mels
        mels = [x[2] for x in batch]
        mel_lens = np.array([mel.shape[0] for mel in mels])
        mels = pad_2D(mels)

        chars = torch.tensor(chars).long()
        labels = torch.tensor(labels).long()
        position = torch.tensor(position).long()
        text_lens = torch.tensor(text_lens).long()
        src_masks = get_mask_from_lengths(text_lens, max(text_lens))
        mels = torch.from_numpy(mels).float()
        mel_lens = torch.tensor(mel_lens).long()
        mel_masks = get_mask_from_lengths(mel_lens, max(mel_lens))

        output = {"mlm_input": chars,
                  "mlm_label": labels,
                  "input_position": position,
                  "text_lens": text_lens,
                  "max_lens": max(text_lens),
                  "src_masks": src_masks,
                  "mels": mels,
                  "mel_lens": mel_lens,
                  "mel_masks": mel_masks
                  }

        return output

if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    import tqdm
    import pandas as pd
    import seaborn as sns  # Python's Statistical Data Visualization Library
    import matplotlib  # for plotting
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "LibriTTS_StyleSpeech_multilingual_diffusion_style_3layer"
    # path = "VNTTS"
    # path = "LibriTTS_StyleSpeech_multilingual_diffusion_style_EN"
    preprocess_config = yaml.load(
        open("./config/config_kaga/{0}/preprocess.yaml".format(path), "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/config_kaga/{0}/train.yaml".format(path), "r"), Loader=yaml.FullLoader
    )
    batch_size = 3
    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataset.collate_fn,
    )
    list_dist = []
    n_batch = 0
    for batchs in tqdm.tqdm(train_loader):
        print(batchs)
        n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in tqdm.tqdm(val_loader):
        n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
