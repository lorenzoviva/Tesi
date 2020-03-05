import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

from tqdm import tqdm
from sklearn.manifold import MDS
from pytorch_extras import RAdam, SingleCycleScheduler
from deps.small_norb.smallnorb.dataset import SmallNORBDataset
from deps.torch_train_test_loop.torch_train_test_loop import LoopComponent, TrainTestLoop

from models import SmallNORBClassifier

DEVICE = 'cuda:0'
# DEVICE = 'cpu:0'

smallnorb = SmallNORBDataset(dataset_root='.data/smallnorb', n_examples=1000)
class SmallNORBTorchDataset(torch.utils.data.Dataset):

    def __init__(self, data, categories, preprocessing):
        self.data = data
        self.categories = categories
        self.preprocess = preprocessing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        images = np.stack((self.data[i].image_lt, self.data[i].image_rt), axis=-1)  # [96, 96, 2]
        images = self.preprocess(images).cuda(device=DEVICE)  # [2, 96, 96]
        category = torch.tensor(self.data[i].category, dtype=torch.long).cuda(device=DEVICE)
        return { 'images': images, 'category': category, }

random_crops = tv.transforms.Compose([
    tv.transforms.ToPILImage(),
    tv.transforms.RandomCrop(size=96, padding=16, padding_mode='edge'),
    tv.transforms.ToTensor()
])

# Normally we would divide train set into train/valid splits; we don't here to match other papers.
trn_ds = SmallNORBTorchDataset(smallnorb.data['train'], smallnorb.categories, random_crops)
val_ds = SmallNORBTorchDataset(smallnorb.data['test'], smallnorb.categories, random_crops)
tst_ds = SmallNORBTorchDataset(smallnorb.data['test'], smallnorb.categories, tv.transforms.ToTensor())

class LoopMain(LoopComponent):

    def __init__(self, n_classes, device, pct_warmup=0.1, mixup=(0.2, 0.2)):
        self.n_classes, self.device, self.pct_warmup = (n_classes, device, pct_warmup)
        self.mixup_dist = torch.distributions.Beta(torch.tensor(mixup[0]), torch.tensor(mixup[1]))
        self.to_onehot = torch.eye(self.n_classes, device=self.device)
        self.saved_data = []

    def on_train_begin(self, loop):
        n_iters = len(loop.train_data) * loop.n_epochs
        loop.optimizer = RAdam(loop.model.parameters(), lr=5e-4)
        loop.scheduler = SingleCycleScheduler(
            loop.optimizer, loop.n_optim_steps, frac=self.pct_warmup, min_lr=1e-5)

    def on_grads_reset(self, loop):
        loop.model.zero_grad()

    def on_forward_pass(self, loop):
        images, category = loop.batch['images'], loop.batch['category']
        target_probs = self.to_onehot[category]

        if loop.is_training:
            r = self.mixup_dist.sample([len(images)]).to(device=images.device)
            idx = torch.randperm(len(images))
            images = images.lerp(images[idx], r[:, None, None, None])
            target_probs = target_probs.lerp(target_probs[idx], r[:, None])

        pred_scores, _, _ = model(images)
        _, pred_ids = pred_scores.max(-1)
        accuracy = (pred_ids == category).float().mean()

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
            'n_samples': len(loop.batch['images']),
            'epoch_desc': loop.epoch_desc,
            'epoch_num': loop.epoch_num,
            'epoch_frac': loop.epoch_num + loop.batch_num / loop.n_batches,
            'batch_num' : loop.batch_num,
            'accuracy': loop.accuracy.item(),
            'loss': loop.loss.item(),
            'lr': loop.optimizer.param_groups[0]['lr'],
            'momentum': loop.optimizer.param_groups[0]['betas'][0],
        })

class LoopProgressBar(LoopComponent):

    def __init__(self, item_names=['loss', 'accuracy']):
        self.item_names = item_names

    def on_epoch_begin(self, loop):
        self.total, self.count = ({ name: 0.0 for name in self.item_names }, 0)
        self.pbar = tqdm(total=loop.n_batches, desc=f"{loop.epoch_desc} epoch {loop.epoch_num}")

    def on_batch_end(self, loop):
        n = len(loop.batch['images'])
        self.count += n
        for name in self.item_names:
            self.total[name] += getattr(loop, name).item() * n
        self.pbar.update(1)
        if (not loop.is_training):
            self.pbar.set_postfix(self.mean)

    def on_epoch_end(self, loop):
        self.pbar.close()

    @property
    def mean(self): return {
        f'mean_{name}': self.total[name] / self.count for name in self.item_names
    }
    # Seed RNG for replicability. Run at least a few times without seeding to measure performance.
# torch.manual_seed(<type an int here>)

# Make iterators for each split, with random shuffling in train set.
trn_itr = torch.utils.data.DataLoader(trn_ds, batch_size=20, shuffle=True)
val_itr = torch.utils.data.DataLoader(val_ds, batch_size=20, shuffle=False)
tst_itr = torch.utils.data.DataLoader(tst_ds, batch_size=20, shuffle=False)

# Initialize model.
n_classes = len(trn_ds.categories)
model = SmallNORBClassifier(n_objs=n_classes, n_parts=64, d_chns=64)
model = model.cuda(device=DEVICE)
print('Total number of parameters: {:,}'.format(sum(np.prod(p.shape) for p in model.parameters())))
# Train model
loop = TrainTestLoop(model, [LoopMain(n_classes, DEVICE), LoopProgressBar()], trn_itr, val_itr)
loop.train(n_epochs=50)
loop.test(tst_itr)

def get_obj_sequence(category, instance, vary='azimuth'):
    const_attr = 'elevation' if (vary == 'azimuth') else 'azimuth'
    samples = [
        s for s in smallnorb.data['test'] if
        (s.category, s.instance, s.lighting, getattr(s, const_attr)) == (category, instance, 0, 0)
    ]
    ds = SmallNORBTorchDataset(samples, smallnorb.categories, tv.transforms.ToTensor())
    with torch.no_grad():
        a, mu, sig2 = loop.model(torch.stack([sample['images'] for sample in ds], dim=0))
    mu = mu[:, category, :, :].cpu()
    return ds.data, mu

data, mu = get_obj_sequence(category=2, instance=0, vary='elevation')
fig, axes = plt.subplots(ncols=len(data), figsize=(len(data), 1))
for sample, axis in zip(data, axes):
    axis.imshow(sample.image_lt, cmap='gray', vmin=0, vmax=255)
    axis.axis('off')
    fig, axis = plt.subplots(figsize=(3, 3))
mds = MDS(n_components=2)
x = mu.contiguous().view(-1, mu.shape[-1]).numpy()
x = mds.fit_transform(x)
x = x.reshape(mu.shape[0], 4, 2)
blue = '#1f77b4'
for i in range(len(x)):
    vert = np.concatenate((x[i], x[i, :1, :]), axis=0)
    alpha = 0.2 + (i + 1.0) / len(x) * 0.8
    axis.plot(vert[:, 0], vert[:, 1], color=blue, alpha=alpha)
    if (i == 0) or (i + 1 == len(x)):
        axis.scatter(vert[0, 0], vert[0, 1], color=blue,
                     facecolor=('white' if i == 0 else blue))
    axis.set(xticks=[], yticks=[])