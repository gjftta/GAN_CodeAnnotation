import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
import logging
from transformers import AdamW, get_linear_schedule_with_warmup
import yaml
from datasets import GDataset
from utils import AttrDict, Batch, Vocab, create_log, load_checkpoint, EarlyStopping
import os
from transformer_model import Seq2SeqTransformer, create_mask, train_epoch, evaluate_loss, evaluate_score

# Data Sourcing&Processing
SRC_LANGUAGE = 'code'
TGT_LANGUAGE = 'docstring'

# Define special symbols
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


class Generator():
    def __init__(self, vocab, config, log):
        self.config = config
        self.log = log
        self.src_vocab_size = vocab.src_vocab_size
        self.tgt_vocab_size = vocab.tgt_vocab_size
        self.code_max_length = config.train.code_max_length
        self.docs_max_length = config.train.docs_max_length
        self.EMB_SIZE = config.EMB_SIZE
        self.NHEAD = config.NHEAD
        self.FFN_HID_DIM = config.FFN_HID_DIM
        self.BATCH_SIZE = config.train.batch_size
        self.NUM_ENCODER_LAYERS = config.NUM_ENCODER_LAYERS
        self.NUM_DECODER_LAYERS = config.NUM_DECODER_LAYERS
        self.best_score = 0
        self.model, self.loss_fn, self.optimizer = self.create_generator()
        self.scheduler = None

    def create_generator(self):
        # Seq2Seq Network using Transformer
        model = Seq2SeqTransformer(self.NUM_ENCODER_LAYERS, self.NUM_DECODER_LAYERS, self.EMB_SIZE, self.NHEAD, self.src_vocab_size, self.tgt_vocab_size, self.FFN_HID_DIM)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.log.debug("Set Generator's parameters: SRC_VOCAB_SIZE:%d TGT_VOCAB_SIZE:%d EMB_SIZE:%d NHEAD:%d FN_HID_DIM:%d BATCH_SIZE:%d NUM_ENCODER_LAYERS:%d NUM_DECODER_LAYERS:%d"
                       % (self.src_vocab_size, self.tgt_vocab_size, self.EMB_SIZE, self.NHEAD, self.FFN_HID_DIM, self.BATCH_SIZE, self.NUM_ENCODER_LAYERS, self.NUM_DECODER_LAYERS))
        self.log.info("Finish creating a Seq2Seq network that uses Transformer")

        # loss_fn = torch.nn.NLLLoss(ignore_index=PAD_IDX)
        loss_fn = torch.nn.NLLLoss()

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.5},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-9)

        self.log.debug("Finish setting parameters&loss&optimizer")
        self.log.info("Finish setting the Generator model")
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

    def batchPGLoss(self, src, tgt, rewards, DEVICE):
        # Returns a policy gradient loss
        src = torch.transpose(src, 0, 1)
        tgt = torch.transpose(tgt, 0, 1)
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE)
        outs = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        outs = torch.log(outs)
        target_onehot = F.one_hot(tgt_output, self.tgt_vocab_size).float()
        # target_onehot[:, :, PAD_IDX] = 0
        # print(target_onehot)
        pred = torch.sum(outs * target_onehot, dim=-1).transpose(0, 1)  # batch_size * seq_len
        loss = -torch.sum(pred * rewards)
        return loss


# Run Generator
if __name__ == "__main__":
    # Load configuration
    config_path = "./configs/generator.yaml"
    config = AttrDict(yaml.load(open(config_path), Loader=yaml.loader.FullLoader))

    # Create log
    log = create_log(config.train.logdir, '/train.log', 'Generator')
    log.info("Start data sourcing&processing...")
    log.debug("Load configuration from %s" % (config_path))

    # Model
    # torch.cuda.set_device(config.DEVICE)
    # print(config.DEVICE)

    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    log.info("Train with device %s" % (DEVICE))

    # Produce random seed (can reproduce)
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # # Load data Iterator
    # Create vocabulary
    vocab = Vocab(GDataset(config.train))

    # Create vocabulary
    generator = Generator(vocab, config, log)

    # if (DEVICE.type == 'cuda') and len(config.DEVICE) > 1:  # multi-GPU
    #     generator.model = torch.nn.DataParallel(generator.model, device_ids=config.DEVICE)

    generator.model = generator.model.to(DEVICE)

    # generator.reload_model()

    early_stop = EarlyStopping(patience=20, verbose=True)

    # Collate function that convert batch of raw strings into batch tensors
    create_batch = Batch(SRC_LANGUAGE, TGT_LANGUAGE)
    create_batch.create_tensor(vocab.vocab_transform)

    # Train&Evaluate
    log_train = logging.getLogger('Train&Evaluate')
    log_train.info("Start training...")

    # Train
    NUM_EPOCHS = config.train.num_epoch
    lossMIN = config.train.lossMIN

    writer = SummaryWriter(config.train.logdir)
    # writer.add_graph(transformer)

    for epoch in range(NUM_EPOCHS):
        start_time = timer()
        # Load train data
        train_iter = GDataset(config.train)
        train_dataloader = DataLoader(train_iter, batch_size=generator.BATCH_SIZE, collate_fn=create_batch.collate_fn_gen)
        t_total = len(train_dataloader) * NUM_EPOCHS
        generator.scheduler = get_linear_schedule_with_warmup(generator.optimizer,
                                                              num_warmup_steps=int(t_total * 0.01),
                                                              num_training_steps=t_total)
        train_loss = train_epoch(generator.model, generator.optimizer, generator.scheduler, generator.loss_fn, train_dataloader, DEVICE, writer, epoch)

        # Load valid data
        val_iter = GDataset(config.valid)
        val_dataloader = DataLoader(val_iter, batch_size=generator.BATCH_SIZE, collate_fn=create_batch.collate_fn_gen)
        val_loss = evaluate_loss(generator.model, generator.loss_fn, val_dataloader, DEVICE)
        writer.add_scalar('val_loss', val_loss, epoch)
        val_score = evaluate_score(generator.model, GDataset(config.valid), generator.docs_max_length + 2, create_batch.text_transform, vocab.vocab_transform, DEVICE)
        writer.add_scalar('val_bleu_1', val_score['Bleu_1'], epoch)
        writer.add_scalar('val_bleu_2', val_score['Bleu_2'], epoch)
        writer.add_scalar('val_bleu_3', val_score['Bleu_3'], epoch)
        writer.add_scalar('val_bleu_4', val_score['Bleu_4'], epoch)
        writer.add_scalar('val_meteor', val_score['METEOR'], epoch)
        writer.add_scalar('val_rouge_l', val_score['ROUGE_L'], epoch)

        end_time = timer()
        log_train.info(f"Epoch: {epoch}, Val loss: {val_loss:.3f}, " f"Epoch time = {(end_time - start_time):.3f}s")
        log_train.info("Val score: Bleu_1: %.3f, Bleu_2: %.3f, Bleu_3: %.3f, Bleu_4: %.3f, METEOR: %.3f, ROUGE_L: %.3f"
                       % (val_score['Bleu_1'], val_score['Bleu_2'], val_score['Bleu_3'], val_score['Bleu_4'], val_score['METEOR'], val_score['ROUGE_L']))

        early_stop(score=val_score['Bleu_4'], gen_model=generator.model, optimizer=generator.optimizer, config=config)
        if early_stop.early_stop:
            if config.train.shuffle is True:
                os.remove(train_iter.shuf_path)
            # os.remove(val_iter.shuf_path)
            log.info("Early stop")
            break

    writer.close()
    log.info("Finish training...")
