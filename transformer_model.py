import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
import math
from timeit import default_timer as timer
import logging
from nlgeval import NLGEval

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int, src_vocab_size: int, tgt_vocab_size: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=2)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor, tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        outs = self.generator(outs)
        return self.softmax(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


# Create Mask
def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, DEVICE):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


# train the model
def train_epoch(model, optimizer, scheduler, loss_fn, train_dataloader, DEVICE, writer, epoch):
    model.train()
    log_train = logging.getLogger('')
    losses = 0

    step = 0
    for src, tgt in train_dataloader:
        # print(src.shape, '\n', tgt.shape)
        start_time = timer()

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        logits_log = torch.log(logits.reshape(-1, logits.shape[-1]))
        loss = loss_fn(logits_log, tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()

        losses += loss.item()

        end_time = timer()
        if step % 100 == 0:
            writer.add_scalar('train_loss', losses / (step + 1), epoch * len(train_dataloader) + step)
            log_train.debug('step: {0}\tloss: {1:.3f}\ttime: {2:.4f}'.format(step, loss.item(), end_time - start_time))
        step += 1

    return losses / len(train_dataloader)


# evaluate model by loss
def evaluate_loss(model, loss_fn, val_dataloader, DEVICE):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        logits_log = torch.log(logits.reshape(-1, logits.shape[-1]))
        loss = loss_fn(logits_log, tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


# evaluate model by typical NLP metrics like bleu
def evaluate_score(model, val_iter, docs_max_length, text_transform, vocab_transform, DEVICE):
    model.eval()
    src_sentences, doc_sentences = zip(*val_iter)
    tgt_sentences, ref_sentences = [], []
    metrics = set(['CIDEr', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])
    nlgeval = NLGEval(metrics_to_omit=metrics)
    for doc_sent in doc_sentences:
        doc_sent = " ".join(doc_sent)
        ref_sentences.append(doc_sent)

    for src_sent in src_sentences:
        tgt_sent = translate(model, src_sent, docs_max_length, text_transform, vocab_transform, DEVICE).strip(' ')
        tgt_sentences.append(tgt_sent)

    ref_list = [ref_sentences]
    score = nlgeval.compute_metrics(ref_list=ref_list, hyp_list=tgt_sentences)

    return score


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, DEVICE):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), DEVICE).type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence, docs_max_length, text_transform, vocab_transform, DEVICE):
    SRC_LANGUAGE = 'code'
    TGT_LANGUAGE = 'docstring'

    # model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=docs_max_length + 2, start_symbol=BOS_IDX, DEVICE=DEVICE).flatten()
    # tgt_tokens = tgt_tokens[1:-1]
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

# # actual function to translate input sentence into target language
# def translate(model: torch.nn.Module, src_sentence, docs_max_length, vocab_transform, DEVICE):
#     # SRC_LANGUAGE = 'code'
#     TGT_LANGUAGE = 'docstring'

#     # model.eval()
#     # src = text_transform[SRC_LANGxsUAGE](src_sentence)
#     # src = src.transposexs(0, 1)
#     src = src_sentence
#     num_tokens = src.shape[0]
#     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
#     tgt_tokens = greedy_decode(model, src, src_mask, max_len=docs_max_length + 2, start_symbol=BOS_IDX, DEVICE=DEVICE).flatten()
#     # tgt_tokens = tgt_tokens[1:-1]
#     return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
