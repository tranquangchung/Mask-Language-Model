import torch.nn as nn
from model.bert import BERT
from model.modules import MelStyleEncoder
import pdb

class BERTLMStyle(nn.Module):
    """
    BERT Language Model
    Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size, model_config):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
        self.melstyle_encoder = MelStyleEncoder(model_config)
        self.init_model()

    def forward(self, x, mask, mels, mel_masks):
        style_vector = self.melstyle_encoder(mels, mel_masks)
        x, attn_list = self.bert(x, style_vector, mask)
        return self.mask_lm(x), attn_list

    def init_model(self):
        un_init = ['bert.embed.weight', 'bert.pos_emb.weight']
        for n, p in self.named_parameters():
            if n not in un_init and p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
