import argparse

import seaborn
import torch
import matplotlib.pyplot as plt
import random
from dataset.vocab import WordVocab
import pdb
from dataset_multi import Dataset
from torch.utils.data import DataLoader
import yaml


def random_word(sentence, vocab):
    tokens = sentence.split()
    tokens_len = [len(token) for token in tokens]
    chars = [char for char in sentence]
    output_label = []

    for i, char in enumerate(chars):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                chars[i] = vocab.mask_index

            # 10% randomly change token to random token
            elif prob < 0.9:
                chars[i] = random.randrange(vocab.vocab_size)

            # 10% randomly change token to current token
            else:
                chars[i] = vocab.char2index(char)

            output_label.append(vocab.char2index(char))

        else:
            chars[i] = vocab.char2index(char)
            output_label.append(0)

    return chars, output_label


def draw(data, x, y, ax):
    seaborn.heatmap(data,
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, # 取值0-1
                    cbar=False, ax=ax)


def Modelload(path):
    assert path is not None
    print(f"path:{path}")
    mlm_encoder = torch.load(path)
    return mlm_encoder


# 验证模型是否收敛
def main():
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("-m", "--model_path", required=True, type=str, help="model of pretrain")
    # parser.add_argument("-v", "--vocab_path", required=True, type=str, help="path of vocab")
    # args = parser.parse_args()
    #
    # model_path = args.model_path
    # vocab_path = args.vocab_path
    #
    # vocab = WordVocab.load_vocab(vocab_path)
    #
    # model = torch.load(model_path,'cpu')
    # model = model.to('cuda')
    # model.eval()
    #
    # sent = '_I _l _o _v _e _C _h _i _n _a _!'.split()
    #
    # text = 'I love China!'
    # sent1, label = random_word(text, vocab)
    # position = [*range(13)]
    # sent1 = torch.tensor(sent1).long().unsqueeze(0).to('cuda')
    # position1 = torch.tensor(position).long().unsqueeze(0).to('cuda')
    # mask_lm_output, attn_list = model.module.forward(sent1, position1)
    # mask_lm_output = torch.argmax(mask_lm_output,dim=2)
    # pdb.set_trace()
    # print(mask_lm_output)

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path", required=True, type=str, help="model of pretrain")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="path of vocab")
    args = parser.parse_args()

    model_path = args.model_path
    vocab_path = args.vocab_path

    model = torch.load(model_path,'cpu')
    model = model.to('cuda')
    model.eval()

    path = "LibriTTS_StyleSpeech_multilingual_diffusion_style_3layer"
    # path = "VNTTS"
    # path = "LibriTTS_StyleSpeech_multilingual_diffusion_style_EN"
    preprocess_config = yaml.load(
        open("./config/config_kaga/{0}/preprocess.yaml".format(path), "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/config_kaga/{0}/train.yaml".format(path), "r"), Loader=yaml.FullLoader
    )

    val_dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False, random=False)
    valid_data_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=val_dataset.collate_fn)
    predict = 0
    total = 0
    for batch in valid_data_loader:
        input_ids = batch["mlm_input"]
        position = batch["input_position"]
        labels = batch["mlm_label"]
        src_masks = batch["src_masks"]
        with torch.no_grad():
            outputs, attn_list = model(input_ids, src_masks)
            outputs = torch.argmax(outputs, dim=2)

        for output, label, input_id in zip(outputs.detach().cpu(), labels.detach().cpu(), input_ids.detach().cpu()):
            mask_index = (input_id == 4).nonzero(as_tuple=True)[0]
            output_index = torch.index_select(output, 0, mask_index)
            label_index = torch.index_select(label, 0, mask_index)
            print("output: ", output_index)
            print("label: ", label_index)
            print("*" * 20)
            predict += torch.eq(output_index, label_index).sum()
            total += label_index.shape[0]
    print(f"True Predict / Total: {predict} / {total}")
    acc = predict / total
    acc = round(acc.item()*100, 3)
    print(f"Acc: {acc} % ")


if __name__ == '__main__':
    main()