#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import getopt
import sys
import os
import math
import time
import argparse
from visdom import Visdom
import tqdm
import data
sys.path.insert(0, os.path.join('..', '..'))
from dnc.dnc import DNC
import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim

from model import DNCModel
from dnc.util import *

parser = argparse.ArgumentParser(description='PyTorch Differentiable Neural Computer')
parser.add_argument('-data', type=str, default='data/penn/', help='location of the data corpus')
parser.add_argument('-bptt', type=int, default=70, help='length of the sequence')

parser.add_argument('-input_size', type=int, default=6, help='dimension of input feature')
parser.add_argument('-rnn_type', type=str, default='lstm', help='type of recurrent cells to use for the controller')
parser.add_argument('-nhid', type=int, default=64, help='number of hidden units of the inner nn')
parser.add_argument('-dropout', type=float, default=0, help='controller dropout')
parser.add_argument('-memory_type', type=str, default='dnc', help='dense or sparse memory: dnc | sdnc | sam')

parser.add_argument('-nlayer', type=int, default=1, help='number of layers')
parser.add_argument('-nhlayer', type=int, default=2, help='number of hidden layers')
parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-optim', type=str, default='adam', help='learning rule, supports adam|rmsprop')
parser.add_argument('-clip', type=float, default=50, help='gradient clipping')

parser.add_argument('-batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('-mem_size', type=int, default=400, help='memory dimension')
parser.add_argument('-mem_slot', type=int, default=16, help='number of memory slots')
parser.add_argument('-read_heads', type=int, default=4, help='number of read heads')
parser.add_argument('-sparse_reads', type=int, default=10, help='number of sparse reads per read head')
parser.add_argument('-temporal_reads', type=int, default=2, help='number of temporal reads')
parser.add_argument('-nr_cells', type=int, default=8, help='Number of memory cells of the DNC / SDNC')

parser.add_argument('-sequence_max_length', type=int, default=4, metavar='N', help='sequence_max_length')
parser.add_argument('-curriculum_increment', type=int, default=0, metavar='N', help='sequence_max_length incrementor per 1K iterations')
parser.add_argument('-curriculum_freq', type=int, default=1000, metavar='N', help='sequence_max_length incrementor per 1K iterations')
parser.add_argument('-cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')

parser.add_argument('-summarize_freq', type=int, default=100, metavar='N', help='summarize frequency')
parser.add_argument('-check_freq', type=int, default=100, metavar='N', help='check point frequency')
parser.add_argument('-visdom', action='store_true', help='plot memory content on visdom per -summarize_freq steps')

parser.add_argument('-emsize', type=int, default=400,help='size of word embeddings')
parser.add_argument('-dropouth', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('-dropouti', type=float, default=0.65, help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('-dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('-wdrop', type=float, default=0.5, help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('-tied', action='store_false', help='tie the word embedding and softmax weights')
parser.add_argument('-debug', action='store_true', help='debug DNC memory contents in visdom (on localhost)')
parser.add_argument('-reset', action='store_true', help='Reset DNC memory contents on every forward pass')
parser.add_argument('-alpha', type=float, default=2, help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('-beta', type=float, default=1, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs to train')

args = parser.parse_args()
print(args)

viz = Visdom()
# assert viz.check_connection()
criterion = nn.CrossEntropyLoss()


if args.cuda != -1:
    print('Using CUDA.')
    T.manual_seed(1111)
else:
    print('Using CPU.')

#
# def repackage_hidden(h):
#     """Wraps hidden states in new Tensors,
#       to detach them from their history."""
#     if isinstance(h, torch.Tensor):
#         return h.detach()
#     elif isinstance(h, (list, )):
#         return [repackage_hidden(v) for v in h]
#     else:
#         return tuple(repackage_hidden(v) for v in h)
#
# def repackage_hidden_dnc(h):
#     if h is None:
#         return None
#
#     if type(h) is list and h[0] is None:
#         return [None]
#
#     (chx, mhxs, _) = h[0]
#     chx = repackage_hidden(chx)
#     if type(mhxs) is list:
#         mhxs = [dict([(k, repackage_hidden(v)) for k, v in mhx.items()]) for mhx in mhxs]
#     else:
#         mhxs = dict([(k, repackage_hidden(v)) for k, v in mhxs.items()])
#     return [(chx, mhxs, None)]


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    elif type(h) is list:
        return [repackage_hidden(v) for v in h]
    elif type(h) is tuple:
        return tuple(repackage_hidden(v) for v in h)


def repackage_hidden_dnc(h):
    if h is None:
        return None

    if type(h) is list and h[0] is None:
        return [None]

    if type(h) is list:
        return [repackage_hidden_dnc(x) for x in h]

    (chx, mhxs, _) = h
    if type(chx) is tuple:
        chx = repackage_hidden_dnc(chx)
    else:
        chx = repackage_hidden(chx)
    if type(mhxs) is list:
        mhxs = [dict([(k, repackage_hidden(v)) for k, v in mhx.items()]) for mhx in mhxs]
    else:
        mhxs = dict([(k, repackage_hidden(v)) for k, v in mhxs.items()])
    return chx, mhxs, None

def stack_arrays(x, y):
    z = np.ndarray([x.shape[0] * x.shape[2] + y.shape[0] * y.shape[2], x.shape[1]])
    for i in range(x.shape[0]):
        start_index = i * (x.shape[2] + y.shape[2])
        z[start_index:start_index + x.shape[2], :] = np.transpose(x[i])
        start_index = start_index + x.shape[2]
        z[start_index:start_index + y.shape[2], :] += np.transpose(y[i])
    return z


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


# def get_batch(source, i, args, seq_len=None, evaluation=False):
#     seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target

def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = torch.from_numpy(np.array(source[i:i+seq_len]))
    target = torch.from_numpy(np.array(source[i+1:i+1+seq_len].view(-1)))
    if args.cuda != -1:
        data = data.cuda()
        target = target.cuda()
    return Variable(data), Variable(target)
# def generate_data(source, i, seq_len, cuda=-1):
#     seq_len = min(seq_len, len(source) - 1 - i)
#     data = torch.from_numpy(np.array(source[i:i+seq_len]))
#     target = torch.from_numpy(np.array(source[i+1:i+1+seq_len].view(-1)))
#     if cuda != -1:
#         data = data.cuda()
#         target = target.cuda()
#     return Variable(data), Variable(target)
#
#
# def batchify(data, bsz, args):
#     # Work out how cleanly we can divide the dataset into bsz parts.
#     nbatch = data.size(0) // bsz
#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.narrow(0, 0, nbatch * bsz)
#     # Evenly divide the data across the bsz batches.
#     data = data.view(bsz, -1).t().contiguous()
#     if args.cuda != -1:
#         data = data.cuda(args.cuda)
#     return data
# criterion = nn.CrossEntropyLoss()


def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = [None]
    hidden = repackage_hidden_dnc(hidden)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden, _ = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden_dnc(hidden)
    return total_loss.item() / len(data_source)


if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname, 'checkpoints')
    if not os.path.isdir(ckpts_dir):
        os.mkdir(ckpts_dir)

    batch_size = args.batch_size
    sequence_max_length = args.sequence_max_length
    summarize_freq = args.summarize_freq
    check_freq = args.check_freq

    corpus = data.Corpus(args.data)
    epoch = 0
    eval_batch_size = 10
    test_batch_size = 1
    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    ntokens = len(corpus.dictionary)

    controllers = None
    mem_slot = args.mem_slot
    mem_size = args.mem_size
    read_heads = args.read_heads

    model = DNCModel(
        ntokens,
        args.emsize,
        args.nhid,
        args.nlayer,
        args.nhlayer,
        args.dropout,
        args.dropouth,
        args.dropouti,
        args.dropoute,
        args.wdrop,
        False,
        args.mem_slot,
        args.read_heads,
        args.sparse_reads,
        args.mem_size,
        args.cuda,
        args.rnn_type,
        controllers
    )
    print(model)
    # register_nan_checks(rnn)

    if args.cuda != -1:
        model.cuda(args.cuda)


    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98])  # 0.0001
    elif args.optim == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98])  # 0.0001
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9, eps=1e-10)  # 0.0001
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)  # 0.01
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    (chx, mhx, rv) = (None, None, None)

    debug_mem = None
    hidden = [None]
    last_save_losses = []
    pbar = tqdm.tqdm(total=train_data.size(0)*args.epochs)
    for epoch in range(args.epochs):
        batch, i = 0, 0
        while i < train_data.size(0) - 1 - 1:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            seq_len = max(5, int(np.random.normal(bptt, 5)))

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            model.train()
            input_data, target_output = get_batch(train_data, i, args, seq_len=seq_len)

            hidden = repackage_hidden_dnc(hidden)

            optimizer.zero_grad()
            output, hidden, rnn_hs, dropped_rnn_hs, debug_mem = model(input_data, hidden, return_h=True, reset_experience=args.reset)

        # loss = criterion(output, target_output)
            #
            # loss.backward()
            #
            # T.nn.utils.clip_grad_norm_(rnn.parameters(), args.clip)
            # optimizer.step()
            # optimizer.step()
            # loss_value = loss.data.item()
            raw_loss = criterion(output.view(-1, ntokens), target_output)

            loss = raw_loss
            # Activiation Regularization
            loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()

            optimizer.step()


            take_checkpoint = (batch != 0) and (batch % check_freq == 0)

            # detach memory from graph ???
            # mhx = {k: (v.detach() if isinstance(v, var) else v) for k, v in mhx.items()}

            last_save_losses.append(loss.data.item())
            summarize = (batch % summarize_freq == 0)
            val_loss2 = evaluate(val_data)

            if summarize:
                loss = np.mean(last_save_losses)  # T.mean(T.stack(last_save_losses))
                if np.isnan(loss):  # T.isnan(loss)
                    raise Exception('nan Loss')

            if summarize and model.debug:
                v = debug_mem
                loss = np.mean(last_save_losses)
                # print(input_data)
                # print("1111111111111111111111111111111111111111111111")
                # print(target_output)
                # print('2222222222222222222222222222222222222222222222')
                # print(F.relu(output))
                last_save_losses = []

                if args.memory_type == 'dnc':
                    viz.heatmap(
                        v[0]['memory'],
                        opts=dict(
                            xtickstep=10,
                            ytickstep=2,
                            title='Memory, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                            ylabel='layer * time',
                            xlabel='mem_slot * mem_size',
                            xmax=1,
                            colormap='Plasma'
                        )
                    )

                viz.heatmap(
                    v[0]['link_matrix'],
                    opts=dict(
                        xtickstep=10,
                        ytickstep=2,
                        title='Link Matrix, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                        ylabel='layer * time',
                        xlabel='mem_slot * mem_slot',
                        xmax=1,
                        colormap='Plasma'
                    )
                )

                viz.heatmap(
                    v[0]['precedence'],
                    opts=dict(
                        xtickstep=10,
                        ytickstep=2,
                        title='Precedence, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                        ylabel='layer * time',
                        xlabel='mem_slot',
                        xmax=1,
                        colormap='Plasma'
                    )
                )

                viz.heatmap(
                    v[0]['read_weights'],
                    opts=dict(
                        xtickstep=10,
                        ytickstep=2,
                        title='Read Weights, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                        ylabel='layer * time',
                        xlabel='nr_read_heads * mem_slot',
                        xmax=1,
                        colormap='Plasma'
                    )
                )

                viz.heatmap(
                    v[0]['write_weights'],
                    opts=dict(
                        xtickstep=10,
                        ytickstep=2,
                        title='Write Weights, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                        ylabel='layer * time',
                        xlabel='mem_slot',
                        xmax=1,
                        colormap='Plasma'
                    )
                )

                viz.heatmap(
                    v[0]['usage_vector'],
                    opts=dict(
                        xtickstep=10,
                        ytickstep=2,
                        title='Usage Vector, t: ' + str(epoch) + ', ppx: ' + str(math.exp(val_loss2)),
                        ylabel='layer * time',
                        xlabel='mem_slot',
                        xmax=1,
                        colormap='Plasma'
                    )
                )
            if take_checkpoint:
                # llprint("\nSaving Checkpoint ... "),
                check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(batch))
                cur_weights = model.state_dict()
                T.save(cur_weights, check_ptr)
                # llprint("Done!\n")
            batch += 1
            i += seq_len
            pbar.set_description_str("\n\ni, batch, epoch: " + str(i) + ", " + str(batch) + ", " + str(epoch) + "\nPerplexity: " + str(math.exp(val_loss2)) + "\nAvg. Logistic Loss: " + str(loss.item()) + "\nCompleated batch size: " + str(input_data.shape) + "\nOn gpu: " + str(input_data.is_cuda)+ "\n")
            pbar.update(seq_len)
    pbar.close()
