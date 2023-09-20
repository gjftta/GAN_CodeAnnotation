import torch
# from timeit import default_timer as timer
from datasets import GDataset
from generate_sample import Sample
from generator import Generator
from discriminator import Discriminator
from rollout import Rollout
from utils import Vocab
from transformers import AdamW


class Model:
    def __init__(self, gen_config, dis_config, gan_config, log):
        self.config = gan_config
        self.log = log
        self.vocab = Vocab(GDataset(gen_config.train))
        self.generator = Generator(self.vocab, gen_config, log)
        # self.optimizer = torch.optim.Adam(self.generator.model.parameters(), lr=0.0001)
        self.optimizer = self.create_optimizer()
        self.scheduler = None
        self.code_max_length = self.generator.code_max_length
        self.docs_max_length = self.generator.docs_max_length
        self.discriminator = Discriminator(self.vocab, dis_config, log)
        self.sample = Sample(self.vocab, self.generator)
        self.BATCH_SIZE = self.config.train.batch_size
        self.SHUFFLE = self.config.train.shuffle
        self.rollout_num = self.config.train.rollout_num
        self.data_path = self.config.train.data_path

    def create_optimizer(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.generator.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.5},
            {'params': [p for n, p in self.generator.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-9)

        return optimizer

    def gan_load_sample(self, data_path=None, DEVICE=None, config=None):
        if data_path is None:
            data_path = self.data_path
        if config is None:
            config = self.config.train

        pos_samples = self.sample.load_sample(data_path, "GAN train positive samples")
        data_iter = GDataset(config)
        # data_iter, len = read_data(data_path, self.code_max_length, self.docs_max_length)
        neg_samples = self.sample.generate_neg_sample(self.generator.model, data_iter, DEVICE, self.BATCH_SIZE)
        return (pos_samples, neg_samples)

    def adv_train_generator_batch(self, src, tgt, DEVICE):
        rollout = Rollout(self.generator, DEVICE)

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        rewards = rollout.get_reward(src, tgt, self.rollout_num, self.discriminator.model)
        adv_loss = self.generator.batchPGLoss(src, tgt, rewards, DEVICE)

        adv_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.generator.model.parameters(), 0.5)

        # self.optimizer.step()
        self.generator.optimizer.step()

        # self.optimizer.zero_grad()
        self.generator.optimizer.zero_grad()

        # self.scheduler.step()
        self.generator.scheduler.step()

        return adv_loss

    def evaluate_loss(self, val_dataloader, DEVICE):
        self.generator.model.eval()
        rollout = Rollout(self.generator, DEVICE)
        losses = 0
        step = 0

        for src, tgt in val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            rewards = rollout.get_reward(src, tgt, self.rollout_num, self.discriminator.model)
            adv_loss = self.generator.batchPGLoss(src, tgt, rewards, DEVICE)

            losses += adv_loss.item()
            step += 1
            if step % 10 == 0:
                return losses / step
