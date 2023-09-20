import torch
from torch.utils.data import DataLoader
import logging
import yaml
import jsonlines
from datasets import GDataset
from generator import Generator
from utils import AttrDict, Batch, read_data
import os
from transformer_model import create_mask, translate
# from generator import Generator
# from utils import Vocab, create_log

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class Sample():
    def __init__(self, vocab, generator: Generator):
        config_path = "./configs/generator.yaml"
        self.config = AttrDict(yaml.load(open(config_path), Loader=yaml.loader.FullLoader))
        # self.config = config
        self.log = logging.getLogger("Sample")
        self.SRC_LANGUAGE, self.TGT_LANGUAGE = 'code', 'docstring'
        self.vocab = vocab
        self.batch = Batch(self.SRC_LANGUAGE, self.TGT_LANGUAGE)
        self.batch.create_tensor(self.vocab.vocab_transform)
        print("Finish creating text_tranform")
        self.model = generator.model
        self.docs_max_length = 32  # 2 is for '<bos>' and '<eos>'

    def load_sample(self, path=None, name="positive samples"):
        # function to load samples already existed.
        if path is None:
            path = self.config.train.data_path
        case, len = read_data(path, self.config.train.code_max_length, 30)
        src_sentences, tgt_sentences = zip(*case)
        src_sentences_tokens, tgt_sentences_tokens = [], []
        for src_sent in src_sentences:
            src_sentences_tokens.append(self.batch.text_transform[self.SRC_LANGUAGE](src_sent))
        for tgt_sent in tgt_sentences:
            tgt_sentences_tokens.append(self.batch.text_transform[self.TGT_LANGUAGE](tgt_sent))
        self.log.info("Finish loading %s...", name)
        return src_sentences_tokens, tgt_sentences_tokens

    def generate_neg_sample_batch(self, model=None, src_sent=None, doc_sent=None, DEVICE=None):
        # Generate negative samples for one batch
        if model is None:
            model = self.model
        if DEVICE is None:
            DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.log.info("Generate with device %s" % (DEVICE))

        model = model.to(DEVICE)
        model.eval()
        src_sent = src_sent.to(DEVICE)
        doc_sent = doc_sent.to(DEVICE)

        tgt_sent = torch.zeros(src_sent.size(1), self.docs_max_length).long()
        tgt_sent[:, 0] = BOS_IDX
        tgt_sent = tgt_sent.to(DEVICE)

        for i in range(1, self.docs_max_length):
            tgt_input = tgt_sent[:, :i].transpose(0, 1)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_sent, tgt_input, DEVICE)
            probs = self.model(src_sent, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)  # prob:batch_size*vocab_size
            # probs = torch.where(torch.isnan(probs), torch.full_like(probs, 0), probs)
            _, next_word = torch.max(probs[-1, :], dim=1)

            tgt_sent[:, i] = next_word

        # self.log.info("Finish generating samples...")

        return src_sent.transpose(0, 1), tgt_sent

    def generate_neg_sample(self, model=None, data_iter=None, DEVICE=None, batch_size=32):
        # if there is not available negative samples, use this funtion to generate.
        if model is None:
            model = self.model
        if DEVICE is None:
            DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.log.info("Generate with device %s" % (DEVICE))
        if data_iter is None:
            data_iter = GDataset(self.config.train)

        model = model.to(DEVICE)
        data_loader = DataLoader(data_iter, batch_size=batch_size, collate_fn=self.batch.collate_fn_gen)
        model.eval()
        src_sentences_tokens = []
        tgt_sentences_tokens = []

        for src_sent, doc_sent in data_loader:
            src_sent = src_sent.to(DEVICE)
            doc_sent = doc_sent.to(DEVICE)

            tgt_sent = torch.zeros(src_sent.size(1), self.docs_max_length).long()
            tgt_sent[:, 0] = BOS_IDX
            tgt_sent = tgt_sent.to(DEVICE)

            for i in range(1, self.docs_max_length):
                tgt_input = tgt_sent[:, :i].transpose(0, 1)
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_sent, tgt_input, DEVICE)
                probs = self.model(src_sent, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)  # prob:batch_size*vocab_size
                # probs = torch.where(torch.isnan(probs), torch.full_like(probs, 0), probs)
                _, next_word = torch.max(probs[-1, :], dim=1)

                tgt_sent[:, i] = next_word

            for i in range(src_sent.size(1)):
                src_sentences_tokens.append(src_sent.transpose(0, 1)[i, :])
                tgt_sentences_tokens.append(tgt_sent[i, :])

        self.log.info("Finish generating samples...")

        return src_sentences_tokens, tgt_sentences_tokens

    def generate_neg_sample_jsonline(self, model=None, data_iter=None, DEVICE=None, output_path=None):
        # function to store generated negative samples in format of '.jsonl'
        if model is None:
            model = self.model
        if DEVICE is None:
            DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.log.info("Generate with device %s" % (DEVICE))
        if data_iter is None:
            data_iter = GDataset(self.config.train)

        model = model.to(DEVICE)
        model.eval()

        if output_path is None:
            output_path = self.config.train.output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path = output_path + "/generated_sample.jsonl"
        with jsonlines.open(output_path, 'w') as file:
            for src_sent, doc_sent in data_iter:
                tgt_sent = translate(model, src_sent, self.docs_max_length, self.batch.text_transform, self.vocab.vocab_transform, DEVICE)
                tgt_sent = tgt_sent.strip().split(' ')
                file.write({"code_tokens": src_sent, "docstring_tokens": tgt_sent, "ture_docstring": doc_sent})
        file.close()
        self.log.info("Finish generating samples with format of '.jsonl'...")

    def generate_neg_sample_txt(self, model=None, data_iter=None, DEVICE=None, output_path=None):
        # function to store generated negative samples in format of '.txt'
        if model is None:
            model = self.model
        if DEVICE is None:
            DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.log.info("Generate with device %s" % (DEVICE))
        if data_iter is None:
            data_iter = GDataset(self.config.train)

        model = model.to(DEVICE)
        model.eval()

        if output_path is None:
            output_path = self.config.train.output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_path1 = output_path + "/reference.txt"
        output_path2 = output_path + "/prediction.txt"
        file1 = open(output_path1, 'w')
        file2 = open(output_path2, 'w')

        n = 0
        for data in data_iter:
            src_sent, doc_sent = data[0], data[1]
            tgt_sent = translate(model, src_sent, self.docs_max_length, self.batch.text_transform, self.vocab.vocab_transform, DEVICE)
            tgt_sent = tgt_sent.strip()
            doc_sent = " ".join(doc_sent)
            file1.write(str(n) + '\t' + doc_sent + '\n')
            file2.write(str(n) + '\t' + tgt_sent + '\n')
            n += 1

        self.log.info("Finish generating samples with format of '.txt'...")
        file1.close()
        file2.close()
