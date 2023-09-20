import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence
import logging
from utils import read_data, shuffle
from timeit import default_timer as timer


class GDataset(IterableDataset):
    # Dataset for Generator
    def __init__(self, config):
        self._logger = logging.getLogger("Dataset")
        self.congif = config
        self.path = config.data_path
        self.code_max_length = config.code_max_length
        self.docs_max_length = config.docs_max_length
        # self.code_vocab = config.code_vocab
        # self.code_vocab_size = config.code_vocab_size
        # self.docstring_vocab = config.docstring_vocab
        # self.docstring_vocab_size = config.docstring_vocab_size
        self.current_pos = None

        self._logger.debug("Load dataset from %s." % self.path)
        if config.shuffle is True:
            start_time = timer()
            self.shuf_path = shuffle(self.path)
            end_time = timer()
            self._logger.debug("Finish shuffling data and restore in %s using time %.3f" % (self.shuf_path, end_time - start_time))
        else:
            self.shuf_path = self.path
        self.cases, self.len = read_data(self.shuf_path, self.code_max_length, self.docs_max_length)
        self._logger.debug("Finish loading dataset with length of %d" % (self.len))
        # self._logger.debug("Load code&docstring vocabularies from %s and %s" % (self.code_vocab, self.docstring_vocab))
        # self.code2idx, self.idx2code = load_vocab(self.code_vocab, self.code_vocab_size)
        # self.docstring2idx, self.idx2docstring = load_vocab(self.docstring_vocab, self.docstring_vocab_size)

        # self._logger.debug("Change sentence into indices according to vocab")
        # self.codes = word_to_indices(self.code2idx, codes)
        # self.docstrings = word_to_indices(self.docstring2idx, docstrings)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.len - 1:
            raise StopIteration
        item = next(self.cases)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.len

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos

    def __str__(self):
        return "GDataset"


class DDataset(IterableDataset):
    # Dataset for Discriminator
    def __init__(self, pos_samples, neg_samples, DEVICE):
        self._logger = logging.getLogger("Dataset")
        # self.config = config
        self.PAD_IDX = 1
        self.neg_source = neg_samples[0]
        self.neg_target = neg_samples[1]
        self.pos_source = pos_samples[0]
        self.pos_target = pos_samples[1]
        self.current_pos = None

        # concatenate neg&pos samples and match them with label
        start_time = timer()

        self.neg_source = pad_sequence(self.neg_source, batch_first=True, padding_value=self.PAD_IDX).to(DEVICE)
        self.neg_target = pad_sequence(self.neg_target, batch_first=True, padding_value=self.PAD_IDX).to(DEVICE)
        self.pos_source = pad_sequence(self.pos_source, batch_first=True, padding_value=self.PAD_IDX).to(DEVICE)
        self.pos_target = pad_sequence(self.pos_target, batch_first=True, padding_value=self.PAD_IDX).to(DEVICE)

        self.src_inp = torch.cat((self.pos_source, self.neg_source), dim=0).long().detach()
        self.tgt_inp = torch.cat((self.pos_target, self.neg_target), dim=0).long().detach()  # !!!need .detach()

        self.label = torch.ones(self.tgt_inp.size(0)).long()
        self.label[self.pos_target.size(0):] = 0  # labels of negative samples is 0
        self.len = self.tgt_inp.size(0)

        end_time = timer()
        self._logger.debug("Finish concatenating and labelling samples with length of %d using time %.3f" % (self.tgt_inp.size(0), end_time - start_time))

        # shuffle
        start_time = timer()
        perm = torch.randperm(self.tgt_inp.size(0))
        self.src_inp = self.src_inp[perm]
        self.tgt_inp = self.tgt_inp[perm]
        self.label = self.label[perm]
        end_time = timer()
        self._logger.debug("Finish shuffling data using time %.3f" % (end_time - start_time))

        self.cases = zip(self.src_inp, self.tgt_inp, self.label)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.len - 1:
            raise StopIteration
        item = next(self.cases)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.len

    def pos(self):
        return self.current_pos

    def __str__(self):
        return "DDataset"


class GANDataset(IterableDataset):
    # Dataset for Discriminator
    def __init__(self, neg_samples):
        self._logger = logging.getLogger("Dataset")
        # self.config = config
        self.PAD_IDX = 1
        self.neg_source = neg_samples[0]
        self.neg_target = neg_samples[1]
        self.current_pos = None

        self.neg_source = pad_sequence(self.neg_source, batch_first=True, padding_value=self.PAD_IDX).long().detach()
        self.neg_target = pad_sequence(self.neg_target, batch_first=True, padding_value=self.PAD_IDX).long().detach()

        self.len = self.neg_source.size(0)

        # shuffle
        start_time = timer()
        perm = torch.randperm(self.neg_source.size(0))
        self.neg_source = self.neg_source[perm]
        self.neg_target = self.neg_target[perm]
        end_time = timer()
        self._logger.debug("Finish shuffling data using time %.3f" % (end_time - start_time))

        self.cases = zip(self.neg_source, self.neg_target)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.len - 1:
            raise StopIteration
        item = next(self.cases)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.len

    def pos(self):
        return self.current_pos

    def __str__(self):
        return "GANDataset"
