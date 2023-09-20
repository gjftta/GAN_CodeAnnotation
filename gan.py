import logging
import random
import torch
import yaml
from CNN_model import evaluate_loss, train_epoch
from transformer_model import evaluate_score
from utils import AttrDict, Batch, EarlyStopping, create_log
from torch.utils.data import DataLoader
from datasets import DDataset, GANDataset, GDataset
from gan_model import Model
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
from transformers import get_linear_schedule_with_warmup


# Data Sourcing&Processing
SRC_LANGUAGE = 'code'
TGT_LANGUAGE = 'docstring'


# Run GAN
if __name__ == "__main__":
    # Load configuration
    dis_config_path = "./configs/discriminator.yaml"
    dis_config = AttrDict(yaml.load(open(dis_config_path), Loader=yaml.loader.FullLoader))

    gen_config_path = "./configs/generator.yaml"
    gen_config = AttrDict(yaml.load(open(gen_config_path), Loader=yaml.loader.FullLoader))

    gan_config_path = "./configs/gan.yaml"
    gan_config = AttrDict(yaml.load(open(gan_config_path), Loader=yaml.loader.FullLoader))

    # Create log
    log = create_log(gan_config.train.logdir, '/train.log', 'GAN')

    log.info("Start data sourcing&processing...")
    log.info("Load configuration from %s" % (gan_config_path))

    print(gan_config.DEVICE)
    DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    log.info("Train with device %s" % (DEVICE))

    # Produce random seed (can reproduce)
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Create vocabulary&G&D
    gan = Model(gen_config, dis_config, gan_config, log)

    # if (DEVICE.type == 'cuda') and len(gan_config.DEVICE) > 1:  # multi-GPU
    #     gan.generator.model = torch.nn.DataParallel(gan.generator.model, device_ids=gan_config.DEVICE)
    #     gan.discriminator.model = torch.nn.DataParallel(gan.discriminator.model, device_ids=gan_config.DEVICE)

    gan.generator.model = gan.generator.model.to(DEVICE)
    gan.discriminator.model = gan.discriminator.model.to(DEVICE)

    gan.generator.reload_model()
    gan.discriminator.reload_model()

    # Set early stop
    gen_early_stop = EarlyStopping(patience=20, verbose=True)
    gen_early_stop.best_score = gan.generator.best_score
    # dis_early_stop = EarlyStopping(patience=20, verbose=True)

    # Collate function that convert batch of raw strings into batch tensors
    create_batch = Batch(SRC_LANGUAGE, TGT_LANGUAGE)
    create_batch.create_tensor(gan.vocab.vocab_transform)

    # Train&Evaluate
    log_train = logging.getLogger('Train&Evaluate')
    log_train.info("Start training...")

    # Train
    NUM_EPOCHS = gan_config.train.num_epoch

    writer = SummaryWriter(gan_config.train.logdir)

    for epoch in range(1):
        # Every epoch train G for 1 times & train D for 3 times
        eval_time = 0
        dis_time = 0
        # Train G
        log_train.info("Start training G...")

        for i in range(NUM_EPOCHS):
            # Load train data
            # start_time = timer()
            # train_samples = gan.gan_load_sample(data_path=gan_config.train.data_path, DEVICE=DEVICE, config=gan_config.train)
            # end_time = timer()
            # log_train.debug(f"Finish generating samples, using time = {(end_time - start_time):.3f}s")

            # train_iter = GANDataset(train_samples[1])
            # train_dataloader = DataLoader(train_iter, batch_size=gan.BATCH_SIZE, shuffle=gan.SHUFFLE)
            train_iter = GDataset(gan_config.train)
            train_dataloader = DataLoader(train_iter, batch_size=gan.BATCH_SIZE, collate_fn=gan.sample.batch.collate_fn_gen)
            t_total = len(train_dataloader) * NUM_EPOCHS
            gan.generator.scheduler = get_linear_schedule_with_warmup(gan.generator.optimizer,
                                                                      num_warmup_steps=int(t_total * 0.01),
                                                                      num_training_steps=t_total)

            # Train
            gan.generator.model.train()
            losses = 0
            step = 0
            start_time = timer()
            # for src, tgt in train_dataloader:
            for src_sent, tgt_sent in train_dataloader:
                if step % 60 == 0 and step != 0:
                    # Train D
                    log_train.info("Start training D...")
                    # dis_early_stop.best_score = 0
                    train_samples = gan.gan_load_sample(data_path=dis_config.train.data_path, DEVICE=DEVICE, config=dis_config.train)
                    val_samples = gan.gan_load_sample(data_path=dis_config.valid.data_path, DEVICE=DEVICE, config=dis_config.valid)
                    for k in range(1):
                        start_time_dis = timer()
                        # Load train data
                        train_iter = DDataset(train_samples[0], train_samples[1], DEVICE)
                        train_dataloader = DataLoader(train_iter, batch_size=gan.discriminator.BATCH_SIZE, shuffle=gan.discriminator.SHUFFLE)
                        # Train
                        adv_loss, train_acc = train_epoch(gan.discriminator.model, gan.discriminator.optimizer, gan.discriminator.loss_fn, train_dataloader, DEVICE, writer, dis_time)

                        # Load valid data
                        val_iter = DDataset(val_samples[0], val_samples[1], DEVICE)
                        val_dataloader = DataLoader(val_iter, batch_size=gan.discriminator.BATCH_SIZE, shuffle=gan.discriminator.SHUFFLE)
                        # Evaluate loss&acc
                        val_loss, val_acc = evaluate_loss(gan.discriminator.model, gan.discriminator.loss_fn, val_dataloader, DEVICE)
                        writer.add_scalar('val_loss', val_loss, dis_time)
                        writer.add_scalar('val_acc', val_acc, dis_time)

                        end_time_dis = timer()
                        log_train.info(f"Epoch: {dis_time}, Val loss: {val_loss:.3f}, Val acc: {val_acc:.3f}," f"Epoch time = {(end_time_dis - start_time_dis):.3f}s")
                        dis_time += 1

                        # dis_early_stop(val_acc, gan.discriminator.model, gan.discriminator.optimizer, dis_config)

                src, tgt = gan.sample.generate_neg_sample_batch(gan.generator.model, src_sent, tgt_sent, DEVICE)
                adv_loss = gan.adv_train_generator_batch(src, tgt, DEVICE)  # Include computing loss and optimizing G
                losses += adv_loss.item()
                # step += 1

                if step % 5 == 0:
                    end_time = timer()
                    writer.add_scalar('train_loss', losses / (step + 1), i * len(train_dataloader) + step)
                    log_train.debug('step: {0}\tloss: {1:.3f}\ttime: {2:.4f}'.format(step, adv_loss.item(), end_time - start_time))
                    start_time = timer()

                if step % 15 == 0:
                    start_time_tot = timer()
                    # Evaluate loss&metics
                    val_score = evaluate_score(gan.generator.model, GDataset(gan_config.valid), gan_config.valid.docs_max_length, create_batch.text_transform, gan.vocab.vocab_transform, DEVICE)
                    writer.add_scalar('val_bleu_1', val_score['Bleu_1'], eval_time)
                    writer.add_scalar('val_bleu_2', val_score['Bleu_2'], eval_time)
                    writer.add_scalar('val_bleu_3', val_score['Bleu_3'], eval_time)
                    writer.add_scalar('val_bleu_4', val_score['Bleu_4'], eval_time)
                    writer.add_scalar('val_meteor', val_score['METEOR'], eval_time)
                    writer.add_scalar('val_rouge_l', val_score['ROUGE_L'], eval_time)

                    log_train.info("Val score: Bleu_1: %.3f, Bleu_2: %.3f, Bleu_3: %.3f, Bleu_4: %.3f, METEOR: %.3f, ROUGE_L: %.3f"
                                   % (val_score['Bleu_1'], val_score['Bleu_2'], val_score['Bleu_3'], val_score['Bleu_4'], val_score['METEOR'], val_score['ROUGE_L']))

                    if step % 15 == 0:
                        # Load valid data
                        val_samples = gan.gan_load_sample(data_path=gan_config.valid.data_path, DEVICE=DEVICE, config=gan_config.valid)
                        val_iter = GANDataset(val_samples[1])
                        val_dataloader = DataLoader(val_iter, batch_size=gan.BATCH_SIZE, shuffle=gan.SHUFFLE)
                        val_loss = gan.evaluate_loss(val_dataloader, DEVICE)
                        writer.add_scalar('val_loss', val_loss, eval_time)
                        end_time_tot = timer()
                        log_train.info(f"Epoch: {eval_time}, Val loss: {val_loss:.3f}, " f"Val time = {(end_time_tot - start_time_tot):.3f}s")

                    eval_time += 1

                    if step % 15 == 0:
                        gen_early_stop(val_score['Bleu_4'], gan.generator.model, gan.discriminator.model, gan.generator.optimizer, gan_config)

                    if gen_early_stop.early_stop:
                        log.info("Early stop")
                        exit(0)

                step += 1
