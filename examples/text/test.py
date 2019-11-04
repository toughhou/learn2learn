#!/usr/bin/env python3

import argparse
import random
import pickle
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

import learn2learn as l2l
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_classes, input_dim=768, inner_dim=200, pooler_dropout=0.3):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = F.log_softmax(self.out_proj(x), dim=1)
        return x


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def compute_loss(task, roberta, learner, loss_func, batch=15):
    loss = 0.0
    acc = 0.0

    for i, (x, y) in enumerate(torch.utils.data.DataLoader(task, batch_size=batch, shuffle=True, num_workers=0)):
        # RoBERTa ENCODING
        x = collate_tokens([roberta.encode(str(sent)) for sent in x], pad_idx=1)    # torch.Size([5, 16])

        with torch.no_grad():  # 网络中的某一个tensor不需要梯度时，可以使用torch.no_grad()来处理
            x = roberta.extract_features(x) # torch.Size([5, 21, 768])  torch.Size([5, 16, 768])

        # 理解：类似bert，取第一位cls代表整个句子表征
        x = x[:, 0, :]  # torch.Size([5, 768])

        # Moving to device
        # x, y = x.to(device), y.view(-1).to(device)
        x = x.to(device)    # (5, 768)
        y = y.view(-1).to(device)   # torch.Size([5])

        output = learner(x)     # torch.Size([5, 5])
        curr_loss = loss_func(output, y)

        acc += accuracy(output, y)
        loss += curr_loss / len(task)

    loss /= len(task)
    return loss, acc


def main(lr=0.005, maml_lr=0.01, iterations=1000, ways=5, shots=1, tps=32, fas=5, device=torch.device("cpu"),
         download_location="meta-data/text"):
    print(device)

    # 保存变量到本地
    text_train = l2l.text.datasets.NewsClassificationDataset(root=download_location, train=True, download=False)
    pickle.dump(text_train, open('text_train.txt', 'wb'))
    pickle.dump(text_train.df_data, open('panda.txt', 'wb'))
    # train_gen = l2l.text.datasets.TaskGenerator(text_train, ways=ways)

    # 保存变量到本地
    train_gen = l2l.data.TaskGenerator(text_train, ways=ways)
    pickle.dump(train_gen, open('train_gen.txt', 'wb'))

    # train_gen = pickle.load(open('train_gen.txt', 'rb'))

    # torch.hub.set_dir(download_location)
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
    roberta.eval()
    roberta.to(device)

    model = Net(num_classes=ways)

    # Put the model to device
    model.to(device)

    # Use parallel in case there are more gpu
    if torch.cuda.device_count() > 1:
        print("Use ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model)

    # Wrap model with MAML
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    loss_func = nn.NLLLoss(reduction="sum")

    tqdm_bar = tqdm(range(iterations))

    # Epoch
    for iteration in tqdm_bar:
        iteration_error = 0.0
        iteration_acc = 0.0

        # Batch(有很多个Task组成）,通过 tasks-per-step控制 - 每个step有多少个task
        for _ in range(tps):
            learner = meta_model.clone()  # 复制一个meta_model，为第一次梯度更新做准备

            # train_task上，task为空，则随机投取N-Way个类
            train_task = train_gen.sample(shots=shots)
            # valid_task上，task为train_task随机抽到的类别，valid_task也需保证使用同样类别
            valid_task = train_gen.sample(shots=shots, task=train_task.sampled_task)

            # Fast Adaptation
            # 作用：为了更好地拟合当前Task。用户同一份数据，做多次模型更新，当step越接近fas时，模型对当前task的拟合越好。
            # fas - steps per fast adaption
            for step in range(fas):
                train_error, _ = compute_loss(train_task, roberta, learner, loss_func, batch=shots * ways)
                learner.adapt(train_error)

            # Compute validation loss
            # 在QuerySet上做更新，使用的模型及参数是在SupportSet上学习得到的cloned model
            valid_error, valid_acc = compute_loss(valid_task, roberta, learner, loss_func, batch=shots * ways)

            iteration_error += valid_error
            iteration_acc += valid_acc

        iteration_error /= tps
        iteration_acc /= tps
        tqdm_bar.set_description("Loss : {:.3f} Acc : {:.3f}".format(iteration_error.item(), iteration_acc))

        # Take the meta-learning step
        # 第2次梯度更新：作用在原始model上
        opt.zero_grad()
        iteration_error.backward()
        opt.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn Text Classification Example')

    parser.add_argument('--ways', type=int, default=5, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=1, metavar='N',
                        help='number of shots (default: 1)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=32, metavar='N',
                        help='tasks per step (default: 32)')
    parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=5, metavar='N',
                        help='steps per fast adaption (default: 5)')

    parser.add_argument('--iterations', type=int, default=100, metavar='N',
                        help='number of iterations (default: 1000)')

    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--maml-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for MAML (default: 0.01)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--download-location', type=str, default="meta-data/text", metavar='S',
                        help='download location for train data and roberta(default : meta-data/text')

    args = parser.parse_args()

    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(lr=args.lr, maml_lr=args.maml_lr, iterations=args.iterations, ways=args.ways, shots=args.shots,
         tps=args.tasks_per_step, fas=args.fast_adaption_steps, device=device,
         download_location=args.download_location)
