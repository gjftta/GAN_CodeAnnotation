import logging
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer as timer


class CNN(nn.Module):
    def __init__(self, emb_size: int, src_vocab_size: int, tgt_vocab_size: int, filter_sizes: list, num_filters: list, dropout: int):
        super(CNN, self).__init__()
        self.feature_dim = sum(num_filters)
        self.emb_size = emb_size
        self.padding_idx = 1
        self.src_embedding = nn.Embedding(src_vocab_size, emb_size, padding_idx=self.padding_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, emb_size, padding_idx=self.padding_idx)
        self.src_convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_size)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.tgt_convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_size)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_tokens: Tensor, tgt_tokens: Tensor):
        tgt_emb = self.tgt_embedding(tgt_tokens.long()).unsqueeze(1)
        src_emb = self.src_embedding(src_tokens.long()).unsqueeze(1)
        src_convs = []
        tgt_convs = []
        # for conv in self.convs:
        #     convs.append(conv(emb))
        src_convs = [conv(src_emb) for conv in self.src_convs]
        tgt_convs = [conv(tgt_emb) for conv in self.tgt_convs]
        src_relus = [F.relu(conv).squeeze(3) for conv in src_convs]
        tgt_relus = [F.relu(conv).squeeze(3) for conv in tgt_convs]
        relus = [torch.cat((src_relu, tgt_relu), dim=2) for src_relu, tgt_relu in zip(src_relus, tgt_relus)]
        pools = [F.max_pool1d(relu, relu.size(2)).squeeze(2) for relu in relus]
        pred = torch.cat(pools, 1)
        highway = self.highway(pred)
        feature = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway
        pred = self.feature2out(self.dropout(feature))

        return pred


# train the model
def train_epoch(model, optimizer, loss_fn, train_dataloader, DEVICE, writer, epoch):
    # model.to(DEVICE)
    model.train()
    log_train = logging.getLogger('')
    losses = 0
    accs = 0
    total_num = 0

    step = 0
    for src_inp, tgt_inp, label in train_dataloader:
        start_time = timer()

        src_inp = src_inp.to(DEVICE)
        tgt_inp = tgt_inp.to(DEVICE)
        label = label.to(DEVICE)
        pred = model.forward(src_inp, tgt_inp)

        optimizer.zero_grad()

        loss = loss_fn(pred, label)
        loss.backward()
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()

        losses += loss.item()
        accs += torch.sum((pred.argmax(dim=-1) == label)).item()
        total_num += tgt_inp.size(0)

        end_time = timer()

        if step % 100 == 0:
            writer.add_scalar('train_loss', losses / (step + 1), epoch * 501 + step)
            writer.add_scalar('train_acc', accs / total_num, epoch * 501 + step)
            log_train.debug('step: {0}\tloss: {1:.3f}\tacc: {2:.3f}\ttime: {3:.4f}'.format(step, loss.item(), accs / total_num, end_time - start_time))
        step += 1

        if step % 500 == 0 and step != 0:
            return losses / (step + 1), accs / total_num


# evaluate model by loss&accuracy
def evaluate_loss(model, loss_fn, val_dataloader, DEVICE):
    model.eval()
    losses = 0
    accs = 0
    total_num = 0

    for src_inp, tgt_inp, tgt in val_dataloader:
        src_inp = src_inp.to(DEVICE)
        tgt_inp = tgt_inp.to(DEVICE)
        tgt = tgt.to(DEVICE)

        pred = model.forward(src_inp, tgt_inp)

        loss = loss_fn(pred, tgt)
        losses += loss.item()
        accs += torch.sum((pred.argmax(dim=-1) == tgt)).item()
        total_num += src_inp.size(0)

    return losses / len(val_dataloader), accs / total_num
