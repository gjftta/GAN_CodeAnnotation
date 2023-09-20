import os
import logging
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from generate_sample import Sample
from generator import Generator
from utils import AttrDict, EarlyStopping, Vocab, create_log, load_checkpoint
from datasets import DDataset, GDataset
from CNN_model import CNN, train_epoch, evaluate_loss
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer

# Data Sourcing&Processing
SRC_LANGUAGE = 'code'
TGT_LANGUAGE = 'docstring'

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


class Discriminator():
    def __init__(self, vocab, config, log):
        self.config = config
        self.log = log
        self.vocab = vocab
        self.src_vocab_size = vocab.src_vocab_size
        self.tgt_vocab_size = vocab.tgt_vocab_size
        self.EMB_SIZE = config.EMB_SIZE
        self.BATCH_SIZE = config.train.batch_size
        self.SHUFFLE = config.train.shuffle
        self.dis_filter_sizes = config.train.filter_sizes
        self.dis_num_filters = config.train.num_filters
        self.dropout = config.train.dropout_rate
        self.model, self.loss_fn, self.optimizer = self.create_discriminator()
        self.train_pos_sample_path = config.train.pos_sample_path
        self.train_neg_sample_path = config.train.neg_sample_path
        self.val_pos_sample_path = config.valid.pos_sample_path
        self.val_neg_sample_path = config.valid.neg_sample_path
        self.best_score = 0

    def create_discriminator(self):
        # Discriminator using CNNs
        model = CNN(self.EMB_SIZE, self.src_vocab_size, self.tgt_vocab_size, self.dis_filter_sizes, self.dis_num_filters, self.dropout)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, a=-0.05, b=0.05)

        self.log.debug("Set Discriminator's parameters: SRC_VOCAB_SIZE:%d TGT_VOCAB_SIZE:%d EMB_SIZE:%d BATCH_SIZE:%d" % (self.src_vocab_size, self.tgt_vocab_size, self.EMB_SIZE, self.BATCH_SIZE))
        self.log.info("Finish creating a Discriminator that uses CNNs")

        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        self.log.debug("Finish setting parameters&loss&optimizer")
        self.log.info("Finish setting the Discriminator model")

        return model, loss_fn, optimizer

    def reload_model(self):
        # Try to reload model checkpoint
        CKPT_path = os.listdir(self.config.train.checkpoint)[0]
        CKPT_path = self.config.train.checkpoint + '/' + CKPT_path
        print(CKPT_path)
        self.model, self.optimizer, self.best_score = load_checkpoint(CKPT_path, self.model, self.optimizer, self.best_score)
        self.log.info("Finish loading model")
        # try:
        #     CKPT_path = os.listdir(self.config.train.checkpoint)[0]
        #     CKPT_path = self.config.train.checkpoint + '/' + CKPT_path
        #     print(CKPT_path)
        #     self.model, self.optimizer = load_checkpoint(CKPT_path, self.model, self.optimizer)
        #     self.log.info("Finish loading model")
        # except:
        #     self.log.info("No checkpoint")

    def dis_load_sample(self, config, generator, train_pos_sample_path=None, train_neg_sample_path=None, val_pos_sample_path=None, val_neg_sample_path=None, DEVICE=None):
        sample = Sample(self.vocab, generator)

        if train_pos_sample_path is None:
            train_pos_sample_path = self.train_pos_sample_path
        if train_neg_sample_path is None:
            train_neg_sample_path = self.train_neg_sample_path
        if val_pos_sample_path is None:
            val_pos_sample_path = self.val_pos_sample_path
        if val_neg_sample_path is None:
            val_neg_sample_path = self.val_neg_sample_path

        # Load negative samples
        # if os.path.exists(train_neg_sample_path):
        #     train_neg_samples = sample.load_sample(train_neg_sample_path, "train negative samples")
        # else:
        #     if not os.path.exists(config.train.neg_sample_dir):
        #         os.makedirs(config.train.neg_sample_dir)
        start_time = timer()
        data_iter = GDataset(self.config.train)
        train_neg_samples = sample.generate_neg_sample(model=generator.model, data_iter=data_iter, DEVICE=DEVICE, batch_size=64)
        end_time = timer()
        self.log.debug(f"Finish generating negative samples for training, using time = {(end_time - start_time):.3f}s")

        # if os.path.exists(val_neg_sample_path):
        #     val_neg_samples = sample.load_sample(val_neg_sample_path, "valid negative samples")
        # else:
        #     if not os.path.exists(config.valid.neg_sample_dir):
        #         os.makedirs(config.valid.neg_sample_dir)
        start_time = timer()
        data_iter = GDataset(self.config.valid)
        val_neg_samples = sample.generate_neg_sample(model=generator.model, data_iter=data_iter, DEVICE=DEVICE, batch_size=64)
        end_time = timer()
        self.log.debug(f"Finish generating negative samples for validing, using time = {(end_time - start_time):.3f}s")

        # Load positive samples
        train_pos_samples = sample.load_sample(train_pos_sample_path, "train positive samples")
        val_pos_samples = sample.load_sample(val_pos_sample_path, "valid positive samples")

        return (train_pos_samples, train_neg_samples), (val_pos_samples, val_neg_samples)


# Run Discriminator
if __name__ == "__main__":
    # Load configuration
    gen_config_path = "./configs/generator.yaml"
    gen_config = AttrDict(yaml.load(open(gen_config_path), Loader=yaml.loader.FullLoader))

    dis_config_path = "./configs/discriminator.yaml"
    dis_config = AttrDict(yaml.load(open(dis_config_path), Loader=yaml.loader.FullLoader))

    # Create log
    log = create_log(dis_config.train.logdir, '/train.log', 'Discriminator')
    log.info("Start data sourcing&processing...")
    log.debug("Load configuration from %s" % (dis_config_path))

    # Model
    # torch.cuda.set_device(config.DEVICE)
    DEVICE = torch.device('cuda:' + str(dis_config.DEVICE) if torch.cuda.is_available() else 'cpu')
    log.info("Train with device %s" % (DEVICE))

    # Produce random seed (can reproduce)
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Create vocabulary
    vocab = Vocab(GDataset(gen_config.train))

    # Create D
    discriminator = Discriminator(vocab, dis_config, log)
    discriminator.model = discriminator.model.to(DEVICE)
    # discriminator.reload_model()

    # Create G and load checkpoint
    generator = Generator(vocab, gen_config, log)
    generator.model = generator.model.to(DEVICE)
    generator.reload_model()

    # Create/Load negative samples
    train_samples, val_samples = discriminator.dis_load_sample(dis_config, generator, DEVICE=DEVICE)

    early_stop = EarlyStopping(patience=5, verbose=True)

    # Train&Evaluate
    log_train = logging.getLogger('Train&Evaluate')
    log_train.info("Start training...")

    # Train
    NUM_EPOCHS = dis_config.train.num_epoch
    lossMIN = dis_config.train.lossMIN

    writer = SummaryWriter(dis_config.train.logdir)
    # writer.add_graph(transformer)

    for epoch in range(NUM_EPOCHS):
        start_time = timer()
        # Load train data
        # train_pos_samples.to(DEVICE)
        train_iter = DDataset(train_samples[0], train_samples[1], DEVICE)
        train_dataloader = DataLoader(train_iter, batch_size=discriminator.BATCH_SIZE, shuffle=discriminator.SHUFFLE)
        train_loss, train_acc = train_epoch(discriminator.model, discriminator.optimizer, discriminator.loss_fn, train_dataloader, DEVICE, writer, epoch)

        # Load valid data
        # val_pos_samples.to(DEVICE)
        val_iter = DDataset(val_samples[0], val_samples[1], DEVICE)
        val_dataloader = DataLoader(val_iter, batch_size=discriminator.BATCH_SIZE, shuffle=discriminator.SHUFFLE)
        val_loss, val_acc = evaluate_loss(discriminator.model, discriminator.loss_fn, val_dataloader, DEVICE)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)

        end_time = timer()
        log_train.info(f"Epoch: {epoch}, Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f}," f"Epoch time = {(end_time - start_time):.3f}s")

        early_stop(score=-val_loss, dis_model=discriminator.model, optimizer=discriminator.optimizer, config=dis_config)
        if early_stop.early_stop:
            log.info("Early stop")
            break
