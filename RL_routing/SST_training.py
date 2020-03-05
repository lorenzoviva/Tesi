import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext as tt

from tqdm import tqdm
from pytorch_extras import RAdam, SingleCycleScheduler
from pytorch_transformers import GPT2Model, GPT2Tokenizer
from deps.torch_train_test_loop.torch_train_test_loop import LoopComponent, TrainTestLoop

from models import SSTClassifier

DEVICE = 'cuda:0'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', do_lower_case=False)
lang_model = GPT2Model.from_pretrained('gpt2-large', output_hidden_states=True, output_attentions=False)
lang_model.cuda(device=DEVICE)
lang_model.eval()
print('Pretrained GPT-2 loaded.')


def tokenized_texts_to_embs(tokenized_texts, pad_token=tokenizer.eos_token):
    tokenized_texts = [[*tok_seq, pad_token] for tok_seq in tokenized_texts]
    lengths = [len(tok_seq) for tok_seq in tokenized_texts]

    max_length = max(lengths)
    input_toks = [t + [pad_token] * (max_length - l) for t, l in zip(tokenized_texts, lengths)]

    input_ids = [tokenizer.convert_tokens_to_ids(tok_seq) for tok_seq in input_toks]
    input_ids = torch.tensor(input_ids).to(device=DEVICE)

    mask = [[1.0] * length + [0.0] * (max_length - length) for length in lengths]
    mask = torch.tensor(mask).to(device=DEVICE)  # [batch sz, num toks]

    with torch.no_grad():
        outputs = lang_model(input_ids=input_ids)
        embs = torch.stack(outputs[-1], -2)  # [batch sz, n toks, n layers, d emb]

    return mask, embs


fine_grained = True  # set to False for binary classification


class SSTFilter():

    def __init__(self, remove_neutral=False, remove_dupes=False):
        self.remove_neutral, self.remove_dupes = (remove_neutral, remove_dupes)
        self.prev_seen = {}

    def __call__(self, sample):
        if self.remove_neutral and (sample.label == 'neutral'):
            return False
        hashable = ''.join(sample.text)
        if self.remove_dupes and (hashable in self.prev_seen):
            return False
        self.prev_seen[hashable] = True
        return True

    tt.datasets.SST.download(root='.data')  # download if necessary


_stoi = {s: i for i, s in enumerate(
    ['very negative', 'negative', 'neutral', 'positive', 'very positive'] \
        if fine_grained else ['negative', 'positive']
)}
TEXT = tt.data.RawField(preprocessing=tokenizer.tokenize, postprocessing=tokenized_texts_to_embs, is_target=False)
LABEL = tt.data.RawField(postprocessing=lambda samples: torch.tensor([_stoi[s] for s in samples], device=DEVICE), is_target=True)

trn_ds = tt.datasets.SST('.data/sst/trees/train.txt', TEXT, LABEL, fine_grained=fine_grained, subtrees=True, filter_pred=SSTFilter(remove_neutral=(not fine_grained), remove_dupes=True))
val_ds = tt.datasets.SST('.data/sst/trees/dev.txt', TEXT, LABEL, fine_grained=fine_grained, subtrees=False, filter_pred=SSTFilter(remove_neutral=(not fine_grained), remove_dupes=False))
tst_ds = tt.datasets.SST('.data/sst/trees/test.txt', TEXT, LABEL, fine_grained=fine_grained, subtrees=False, filter_pred=SSTFilter(remove_neutral=(not fine_grained), remove_dupes=False))

print('Datasets ready.')
print('Number of samples: {:,} train phrases, {:,} valid sentences, {:,} test sentences.' \
      .format(len(trn_ds), len(val_ds), len(tst_ds)))


class LoopMain(LoopComponent):

    def __init__(self, n_classes, device, pct_warmup=0.1, mixup=(0.2, 0.2)):
        self.n_classes, self.device, self.pct_warmup = (n_classes, device, pct_warmup)
        self.mixup_dist = torch.distributions.Beta(torch.tensor(mixup[0]), torch.tensor(mixup[1]))
        self.onehot = torch.eye(self.n_classes, device=self.device)
        self.saved_data = []

    def on_train_begin(self, loop):
        n_iters = len(loop.train_data) * loop.n_epochs
        loop.optimizer = RAdam(loop.model.parameters(), lr=5e-4)
        loop.scheduler = SingleCycleScheduler(
            loop.optimizer, loop.n_optim_steps, frac=self.pct_warmup, min_lr=1e-5)

    def on_grads_reset(self, loop):
        loop.model.zero_grad()

    def on_forward_pass(self, loop):
        model, batch = (loop.model, loop.batch)
        mask, embs = batch.text
        target_probs = self.onehot[batch.label]

        if loop.is_training:
            r = self.mixup_dist.sample([len(mask)]).to(device=mask.device)
            idx = torch.randperm(len(mask))
            mask = mask.lerp(mask[idx], r[:, None])
            embs = embs.lerp(embs[idx], r[:, None, None, None])
            target_probs = target_probs.lerp(target_probs[idx], r[:, None])

        pred_scores, _, _ = model(mask, embs)
        _, pred_ids = pred_scores.max(-1)
        accuracy = (pred_ids == batch.label).float().mean()

        loop.pred_scores, loop.target_probs, loop.accuracy = (pred_scores, target_probs, accuracy)

    def on_loss_compute(self, loop):
        losses = -loop.target_probs * F.log_softmax(loop.pred_scores, dim=-1)  # CE
        loop.loss = losses.sum(dim=-1).mean()  # sum of classes, mean of batch

    def on_backward_pass(self, loop):
        loop.loss.backward()

    def on_optim_step(self, loop):
        loop.optimizer.step()
        loop.scheduler.step()

    def on_batch_end(self, loop):
        self.saved_data.append({
            'n_samples': len(loop.batch),
            'epoch_desc': loop.epoch_desc,
            'epoch_num': loop.epoch_num,
            'epoch_frac': loop.epoch_num + loop.batch_num / loop.n_batches,
            'batch_num': loop.batch_num,
            'accuracy': loop.accuracy.item(),
            'loss': loop.loss.item(),
            'lr': loop.optimizer.param_groups[0]['lr'],
            'momentum': loop.optimizer.param_groups[0]['betas'][0],
        })


class LoopProgressBar(LoopComponent):

    def __init__(self, item_names=['loss', 'accuracy']):
        self.item_names = item_names

    def on_epoch_begin(self, loop):
        self.total, self.count = ({name: 0.0 for name in self.item_names}, 0)
        self.pbar = tqdm(total=loop.n_batches, desc=f"{loop.epoch_desc} epoch {loop.epoch_num}")

    def on_batch_end(self, loop):
        n = len(loop.batch)
        self.count += n
        for name in self.item_names:
            self.total[name] += getattr(loop, name).item() * n
        self.pbar.update(1)
        if (not loop.is_training):
            means = {f'mean_{name}': self.total[name] / self.count for name in self.item_names}
            self.pbar.set_postfix(means)

    def on_epoch_end(self, loop):
        self.pbar.close()
        # Seed RNG for replicability. Run at least a few times without seeding to measure performance.


# torch.manual_seed(<type an int here>)

# Make iterators for each split.
trn_itr, val_itr, tst_itr = tt.data.Iterator.splits(
    (trn_ds, val_ds, tst_ds),
    shuffle=True,
    batch_size=20,
    device=DEVICE)

# Initialize model.
n_classes = len(_stoi)
model = SSTClassifier(
    d_depth=lang_model.config.n_layer + 1,
    d_emb=lang_model.config.hidden_size,
    d_inp=64,
    d_cap=2,
    n_parts=64,
    n_classes=n_classes,
)
model = model.cuda(device=DEVICE)
print('Total number of parameters: {:,}'.format(sum(np.prod(p.shape) for p in model.parameters())))
# Train model
loop = TrainTestLoop(model, [LoopMain(n_classes, DEVICE), LoopProgressBar()], trn_itr, val_itr)
loop.train(n_epochs=3)
loop.test(tst_itr)
