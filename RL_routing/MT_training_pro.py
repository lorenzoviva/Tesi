from typing import Tuple

from torch import Tensor
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dnc.dnc import DNC
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import math
import time

class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()

class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        # self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.rnn = SSTClassifier(
            d_depth=enc_hid_dim,
            d_emb=emb_dim,
            d_inp=enc_hid_dim,
            d_cap=2,
            n_parts=enc_hid_dim,
            n_classes=output_dim)
        #(enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0)

class SSTClassifier(nn.Module):
    """
    Args:
        d_depth: int, number of embeddings per token.
        d_emb: int, dimension of token embeddings.
        d_inp: int, number of features computed per embedding.
        d_cap: int, dimension 2 of output capsules.
        n_parts: int, number of parts detected.
        n_classes: int, number of classes.

    Input:
        mask: [..., n] tensor with 1.0 for tokens, 0.0 for padding.
        embs: [..., n, d_depth, d_emb] embeddings for n tokens.

    Output:
        a_out: [..., n_classes] class scores.
        mu_out: [..., n_classes, 1, d_cap] class capsules.
        sig2_out: [..., n_classes, 1, d_cap] class capsule variances.
    """
    def __init__(self, d_depth, d_emb, d_inp, d_cap, n_parts, n_classes):
        super().__init__()
        self.depth_emb = nn.Parameter(torch.zeros(d_depth, d_emb))
        self.detect_parts = nn.Sequential(nn.Linear(d_emb, d_inp), Swish(), nn.LayerNorm(d_inp))
        self.routings = nn.Sequential(
            Routing(d_cov=1, d_inp=d_inp, d_out=d_cap, n_out=n_parts),
            Routing(d_cov=1, d_inp=d_cap, d_out=d_cap, n_inp=n_parts, n_out=n_classes),
        )
        nn.init.kaiming_normal_(self.detect_parts[0].weight)
        nn.init.zeros_(self.detect_parts[0].bias)

    def forward(self, mask, embs):
        a = torch.log(mask / (1.0 - mask))                     # -inf to inf (logit)
        a = a.unsqueeze(-1).expand(-1, -1, embs.shape[-2])     # [bs, n, d_depth]
        a = a.contiguous().view(a.shape[0], -1)                # [bs, (n * d_depth)]

        mu = self.detect_parts(embs + self.depth_emb)          # [bs, n, d_depth, d_inp]
        mu = mu.view(mu.shape[0], -1, 1, mu.shape[-1])         # [bs, (n * d_depth), 1, d_inp]

        for routing in self.routings:
            a, mu, sig2 = routing(a, mu)

        return a, mu, sig2

class Routing(nn.Module):
    """
    Official implementation of the routing algorithm proposed by "An
    Algorithm for Routing Capsules in All Domains" (Heinsen, 2019),
    https://arxiv.org/abs/1911.00792.

    Args:
        d_cov: int, dimension 1 of input and output capsules.
        d_inp: int, dimension 2 of input capsules.
        d_out: int, dimension 2 of output capsules.
        n_inp: (optional) int, number of input capsules. If not provided, any
            number of input capsules will be accepted, limited by memory.
        n_out: (optional) int, number of output capsules. If not provided, it
            will be equal to the number of input capsules, limited by memory.
        n_iters: (optional) int, number of routing iterations. Default is 3.
        single_beta: (optional) bool, if True, beta_use and beta_ign are the
            same parameter, otherwise they are distinct. Default: False.
        eps: (optional) small positive float << 1.0 for numerical stability.

    Input:
        a_inp: [..., n_inp] input scores
        mu_inp: [..., n_inp, d_cov, d_inp] capsules of shape d_cov x d_inp

    Output:
        a_out: [..., n_out] output scores
        mu_out: [..., n_out, d_cov, d_out] capsules of shape d_cov x d_out
        sig2_out: [..., n_out, d_cov, d_out] variances of shape d_cov x d_out

    Sample usage:
        >>> a_inp = torch.randn(100)  # 100 input scores
        >>> mu_inp = torch.randn(100, 4, 4)  # 100 capsules of shape 4 x 4
        >>> m = Routing(d_cov=4, d_inp=4, d_out=4, n_inp=100, n_out=10)
        >>> a_out, mu_out, sig2_out = m(a_inp, mu_inp)
        >>> print(mu_out)  # 10 capsules of shape 4 x 4
    """
    def __init__(self, d_cov, d_inp, d_out, n_inp=-1, n_out=-1, n_iters=3, single_beta=False, eps=1e-5):
        super().__init__()
        self.n_iters, self.eps = (n_iters, eps)
        self.n_inp_is_fixed, self.n_out_is_fixed = (n_inp > 0, n_out > 0)
        one_or_n_inp, one_or_n_out = (max(1, n_inp), max(1, n_out))
        self.register_buffer('CONST_one', torch.tensor(1.0))
        self.W = nn.Parameter(torch.empty(one_or_n_inp, one_or_n_out, d_inp, d_out).normal_() / d_inp)
        self.B = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out, d_cov, d_out))
        self.beta_use = nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.beta_ign = self.beta_use if single_beta else nn.Parameter(torch.zeros(one_or_n_inp, one_or_n_out))
        self.f = nn.Sigmoid()
        self.log_f = nn.LogSigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, a_inp, mu_inp):
        n_inp = a_inp.shape[-1]
        n_out = self.W.shape[1] if self.n_out_is_fixed else n_inp
        W = self.W
        W = W if self.n_inp_is_fixed else W.expand(n_inp, -1, -1, -1)
        W = W if self.n_out_is_fixed else W.expand(-1, n_out, -1, -1)
        V = torch.einsum('ijdh,...icd->...ijch', W, mu_inp) + self.B
        for iter_num in range(self.n_iters):

            # E-step.
            if iter_num == 0:
                R = (self.CONST_one / n_out).expand(V.shape[:-2])  # [...ij]
            else:
                log_p_simplified = \
                    - torch.einsum('...ijch,...jch->...ij', V_less_mu_out_2, 1.0 / (2.0 * sig2_out)) \
                    - sig2_out.sqrt().log().sum((-2, -1)).unsqueeze(-2)
                R = self.softmax(self.log_f(a_out).unsqueeze(-2) + log_p_simplified)  # [...ij]

            # D-step.
            f_a_inp = self.f(a_inp).unsqueeze(-1)  # [...i1]
            D_use = f_a_inp * R
            D_ign = f_a_inp - D_use

            # M-step.
            a_out = (self.beta_use * D_use).sum(dim=-2) - (self.beta_ign * D_ign).sum(dim=-2)  # [...j]
            over_D_use_sum = 1.0 / (D_use.sum(dim=-2) + self.eps)  # [...j]
            mu_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V, over_D_use_sum)
            V_less_mu_out_2 = (V - mu_out.unsqueeze(-4)) ** 2  # [...ijch]
            sig2_out = torch.einsum('...ij,...ijch,...j->...jch', D_use, V_less_mu_out_2, over_D_use_sum) + self.eps

        return a_out, mu_out, sig2_out

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SRC = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = "spacy",
            tokenizer_language="en",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adam(model.parameters())


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

PAD_IDX = TRG.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

N_EPOCHS = 10
CLIP = 1

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')