from module import optim4GPU
import seaborn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from model import BERTLM, BERT, BERTLMTTS
from transformer import Encoder
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import configs.hparams as hp
import sys
import traceback
import pdb


class BERTTrainerTTS:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: Encoder, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = hp.log_freq, args=None, global_step=0, path=None):
        """
        :param bert: MLM model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        self.args = args
        self.step = global_step
        self.path = path

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLMTTS(bert, vocab_size).to(self.device)
        print(self.model)
        # load pretrained model
        # path = "/home/s2220411/Code/new_explore/Mask-Language-Model/output/model_mlm/mlm_ep499.model"
        # self.mlm_encoder = torch.load(path).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            # self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        total_steps = hp.epochs * len(self.train_data)
        self.optimer = optim4GPU(self.model, total_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        # Writer
        self.log_freq = log_freq
        # train
        self.train_loss_writer = SummaryWriter(f'{self.path.runs_path}/train/train_loss')
        self.train_model_param_writer = SummaryWriter(f'{self.path.runs_path}/train/model_param')
        # valid
        self.valid_loss_writer = SummaryWriter(f'{self.path.runs_path}/valid/valid_loss')

        self.num_params()

    def train(self):

        train_writer = [self.train_loss_writer, self.train_model_param_writer]
        valid_writer = [self.valid_loss_writer]
        try:
            for epoch in range(hp.epochs):

                # Setting the tqdm progress bar
                data_iter = tqdm.tqdm(enumerate(self.train_data),
                                      desc="EP_%s:%d" % ("train", epoch),
                                      total=len(self.train_data),
                                      bar_format="{l_bar}{r_bar}")

                running_loss = 0
                for i, data in data_iter:

                    self.step += 1

                    # 0. batch_data will be sent into the device(GPU or cpu)
                    data = {key: value.to(self.device) for key, value in data.items()}

                    # 1. forward masked_lm model
                    mask_lm_output, attn_list = self.model.forward(data["mlm_input"], data["src_masks"])
                    # with torch.no_grad():
                    #     mask_1, attn1 = self.mlm_encoder(data["mlm_input"], data["input_position"])

                    # 2. NLLLoss of predicting masked token word
                    self.optimer.zero_grad()
                    loss = self.criterion(mask_lm_output.transpose(1, 2), data["mlm_label"])

                    # 3. backward and optimization only in train
                    loss.backward()
                    self.optimer.step()

                    # loss
                    running_loss += loss.item()
                    avg_loss = running_loss / (i + 1)

                    # write log
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "step": self.step,
                        "avg_loss": avg_loss,
                        "loss": loss.item()
                    }
                    if i % self.log_freq == 0:
                        data_iter.write(str(post_fix))

                    # writer train loss
                    if self.step % hp.save_train_loss == 0:
                        self.train_loss_writer.add_scalar('train_loss', loss, self.step)

                    # save bert model
                    if self.step % hp.save_model == 0:
                        self.save_model(epoch, f"{self.path.bert_path}/bert")
                        self.save_mlm_model(epoch, f"{self.path.mlm_path}/mlm")

                    # evaluate
                    if self.step % hp.save_valid_loss == 0:
                        valid_loss = self.evaluate(epoch, valid_writer)


                valid_loss = self.evaluate(epoch, valid_writer)
                print(f"EP_{epoch}, train_avg_loss={avg_loss}, valid_avg_loss={valid_loss}")

            for writer in train_writer:
                writer.close()
            for writer in valid_writer:
                writer.close()
            if True:
                self.save_model(epoch, f"{self.path.bert_path}/bert")
                self.save_mlm_model(epoch, f"{self.path.mlm_path}/mlm")

        except BaseException:
            traceback.print_exc()
            for writer in train_writer:
                writer.close()
            for writer in valid_writer:
                writer.close()



    def evaluate(self, epoch, valid_writer):
        [self.valid_loss_writer] = valid_writer
        self.model.eval()

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.test_data),
                              desc="EP_%s:%d" % ("test", epoch),
                              total=len(self.test_data),
                              bar_format="{l_bar}{r_bar}")

        running_loss = 0
        with torch.no_grad():
            for i, data in data_iter:

                self.step += 1

                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward masked_lm model
                mask_lm_output, attn_list = self.model.forward(data["mlm_input"], data["src_masks"])

                # 2. NLLLoss of predicting masked token word
                loss = self.criterion(mask_lm_output.transpose(1, 2), data["mlm_label"])

                # loss
                running_loss += loss.cpu().detach().numpy()
                avg_loss = running_loss / (i + 1)

                # print log
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "step": self.step,
                    "avg_loss": avg_loss,
                    "loss": loss.item()
                }
                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

                # writer valid loss
                self.valid_loss_writer.add_scalar('valid_loss', loss, self.step)

            print(f"Valid Over!")
            return avg_loss


    def stream(self, message):
        sys.stdout.write(f"\r{message}")

    def draw(self, data, x, y, ax):
        seaborn.heatmap(data,
                        xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,  # 取值0-1
                        cbar=False, ax=ax)

    def num_params(self, print_out=True):
        params_requires_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        params_requires_grad = sum([np.prod(p.size()) for p in params_requires_grad]) #/ 1_000_000

        parameters = sum([np.prod(p.size()) for p in self.model.parameters()]) #/ 1_000_000
        if print_out:
            print('Trainable total Parameters: %d' % parameters)
            print('Trainable requires_grad Parameters: %d' % params_requires_grad)


    def save_model(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "_ep%d.model" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def save_mlm_model(self, epoch, file_path="output/mlm_trained.model"):
        """
        Saving the current MLM model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "_ep%d.model" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path








