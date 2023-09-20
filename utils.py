import jsonlines
import os
import random
import codecs
import numpy as np
import torch
from tempfile import mkstemp
import logging
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from typing import Iterable, List


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


def create_log(path, log_name, model_name):
    # Create log
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=path + log_name, level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    log = logging.getLogger(model_name)
    return log


def read_data(path, code_max_length, docs_max_length):
    codes, docstrings = [], []
    with jsonlines.open(path) as data:
        for sent in data:
            # print(sent['code_tokens'], "\n", sent['docstring_tokens'])
            code = sent['code_tokens']
            docstring = sent['docstring_tokens']
            # print(len(src_sent), ' ', len(dst_sent), "\n", src_sent, "\n", dst_sent)
            if len(code) > code_max_length or len(docstring) > docs_max_length:
                continue
            codes.append(code)  # 处理过的sentence pair放入容器
            docstrings.append(docstring)
            # Create a padded batch.
    return zip(codes, docstrings), len(codes)


def shuffle(file):
    tf_os, tpath = mkstemp()  # create a temp file
    tf = open(tpath, 'w')

    print(file)
    fds = open(file)

    for lines in fds:
        tf.writelines(lines)

    fds.close()
    tf.close()

    os.system('shuf %s > %s' % (tpath, tpath + '.shuf'))

    fds = open(file + '.{}.shuf'.format(os.getpid()), 'w')

    tf = open(tpath + '.shuf', 'r')
    lines = tf.readlines()
    random.shuffle(lines)

    for i in lines:
        fds.writelines(i)

    fds.close()

    os.remove(tpath)
    os.remove(tpath + '.shuf')

    return file + '.{}.shuf'.format(os.getpid())


def load_vocab(path, vocab_size):
    # Use this function to load vocabularies which can be directly used
    vocab = [line.split("\n")[0] for line in codecs.open(path, 'r', 'utf-8')]
    vocab = vocab[:vocab_size]
    assert len(vocab) == vocab_size
    word2idx = {word: idx for idx, word in enumerate(vocab)}  # 两种对应关系，word和idx分别为key
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


class Vocab():
    """
    Use this Class to create vocabularies with dataset if there are not vocabularies which can be directly used.
    """
    def __init__(self, data_iter):
        self.UNK_IDX = 0
        self.data_iter = data_iter
        self.log = logging.getLogger("Vocabulary")
        self.token_transform = {}
        self.src_language_size = 0
        self.tgt_language_size = 0
        self.src_vocab_size = 0
        self.tgt_vocab_size = 0
        self.vocab_transform = self.create_voc('code', 'docstring')

    def create_voc(self, SRC_LANGUAGE, TGT_LANGUAGE):
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        vocab_transform = {}
        self.token_transform[SRC_LANGUAGE], self.token_transform[TGT_LANGUAGE] = zip(*self.data_iter)
        self.src_language_size, self.tgt_language_size = len(self.token_transform[SRC_LANGUAGE]), len(self.token_transform[TGT_LANGUAGE])

        self.log.info("Finish tokenizing: code_token_size:%d docstring_token_size:%d" % (self.src_language_size, self.tgt_language_size))

        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            # Create torchtext's Vocab object
            vocab_transform[ln] = build_vocab_from_iterator(self.yield_tokens(self.token_transform[ln]), min_freq=1, specials=special_symbols, special_first=True)

        self.src_vocab_size, self.tgt_vocab_size = len(vocab_transform[SRC_LANGUAGE]), len(vocab_transform[TGT_LANGUAGE])
        self.log.info("Finish creating vocabularies\nVocabulary of code has %d words & Vocabulary of docstring has %d words" % (self.src_vocab_size, self.tgt_vocab_size))
        # Set UNK_IDX as the default index. This index is returned when the token is not found.
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            vocab_transform[ln].set_default_index(self.UNK_IDX)
        return vocab_transform

    def yield_tokens(self, data_iter: Iterable) -> List[str]:
        for data_sample in data_iter:
            yield data_sample


def word_to_indices(word2idx, sents):
    # Convert words to indices.
    indices = []
    for sent in sents:
        x = [word2idx.get(word, 1) for word in (sent + [u"</S>"])]  # 1: OOV, </S>: End of Text
        indices.append(x)

    # Pad to the same length.
    maxlen = max([len(s) for s in indices])
    X = np.zeros([len(indices), maxlen], np.int32)
    for i, x in enumerate(indices):
        X[i, :len(x)] = x

    return X


class Batch():
    '''
    class to collate data samples into batch tesors
    '''
    def __init__(self, SRC_LANGUAGE, TGT_LANGUAGE):
        # Define special symbols and indices
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        self.SRC_LANGUAGE = SRC_LANGUAGE
        self.TGT_LANGUAGE = TGT_LANGUAGE
        self.text_transform = {}

    # function to collate data samples into batch tesors
    def collate_fn_gen(self, batch):
        src_batch, tgt_batch = [], []

        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[self.SRC_LANGUAGE](src_sample))
            tgt_batch.append(self.text_transform[self.TGT_LANGUAGE](tgt_sample))

        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)

        return src_batch, tgt_batch

    def collate_fn_dis(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[self.TGT_LANGUAGE](src_sample))
            tgt_batch.append(tgt_sample)

        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        return src_batch, tgt_batch

    def create_tensor(self, vocab_transform):
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            self.text_transform[ln] = self.sequential_transforms(
                # token_transform[ln],  # Tokenization
                vocab_transform[ln],  # Numericalization
                self.tensor_transform)  # Add BOS/EOS and create tensor

    # helper function to club together sequential operations
    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input

        return func

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.BOS_IDX]), torch.tensor(token_ids), torch.tensor([self.EOS_IDX])))


def load_checkpoint(checkpoint_PATH=None, model=None, optimizer=None, best_score=None):
    if checkpoint_PATH is not None:
        model_CKPT = torch.load(checkpoint_PATH, map_location=lambda storage, loc: storage.cuda(2))
        if optimizer is not None:
            optimizer.load_state_dict(model_CKPT['optimizer'])
        if model is not None:
            model.load_state_dict(model_CKPT['state_dict'])
        if best_score is not None:
            best_score = model_CKPT['best_score']
    if model is not None and optimizer is not None and best_score is not None:
        return model, optimizer, best_score
    elif model is not None and best_score is not None:
        return model, best_score
    else:
        return model


def gan_load_checkpoint(checkpoint_PATH=None, gen_model=None, dis_model=None, gen_optimizer=None, best_score=None):
    if checkpoint_PATH is not None:
        model_CKPT = torch.load(checkpoint_PATH, map_location=lambda storage, loc: storage.cuda(1))
        if gen_optimizer is not None:
            gen_optimizer.load_state_dict(model_CKPT['optimizer'])
        if gen_model is not None:
            gen_model.load_state_dict(model_CKPT['gen_state_dict'])
        if dis_model is not None:
            dis_model.load_state_dict(model_CKPT['dis_state_dict'])
        if best_score is not None:
            best_score = model_CKPT['best_score']
    if gen_model is not None and dis_model is not None and gen_optimizer is not None and best_score is not None:
        return gen_model, dis_model, gen_optimizer, best_score
    elif gen_model is not None and dis_model is not None and best_score is not None:
        return gen_model, dis_model, best_score
    else:
        return gen_model, dis_model


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, score, gen_model=None, dis_model=None, optimizer=None, config=None):

        # score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, gen_model, dis_model, optimizer, config)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, gen_model, dis_model, optimizer, config)
            self.counter = 0

    def save_checkpoint(self, score, gen_model, dis_model, optimizer, config):
        '''Saves model when validation loss decrease.'''
        log = logging.getLogger('')
        if self.verbose:
            log.info(f'Validation accuracy increased ({self.val_loss_min:.6f} --> {score:.6f}).  Saving model ...')
        if not os.path.exists(config.train.checkpoint):
            os.makedirs(config.train.checkpoint)
        if len(os.listdir(config.train.checkpoint)) > 0:
            os.remove(config.train.checkpoint + '/' + os.listdir(config.train.checkpoint)[0])
        if dis_model is None:
            torch.save({'state_dict': gen_model.state_dict(), 'best_score': score, 'optimizer': optimizer.state_dict()},
                       config.train.checkpoint + '/checkpoint-' + str("%.4f" % score) + '.pth.tar')  # 这里会存储迄今最优模型的参数
        elif gen_model is None:
            torch.save({'state_dict': dis_model.state_dict(), 'best_score': score, 'optimizer': optimizer.state_dict()},
                       config.train.checkpoint + '/checkpoint-' + str("%.4f" % score) + '.pth.tar')  # 这里会存储迄今最优模型的参数
        else:
            torch.save({'gen_state_dict': gen_model.state_dict(), 'dis_state_dict': dis_model.state_dict(), 'best_score': score, 'optimizer': optimizer.state_dict()},
                       config.train.checkpoint + '/checkpoint-' + str("%.4f" % score) + '.pth.tar')  # 这里会存储迄今最优模型的参数
        # torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = score


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total, used


def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
