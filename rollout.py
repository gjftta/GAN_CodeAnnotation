import torch
import torch.nn.functional as F
from transformer_model import create_mask


class Rollout:
    def __init__(self, generator, DEVICE):
        self.generator = generator
        self.model = generator.model
        self.docs_max_length = generator.docs_max_length + 2    # 2 is for '<bos>' and '<eos>'
        self.src_vocab_size = generator.src_vocab_size
        self.tgt_vocab_size = generator.tgt_vocab_size
        self.DEVICE = DEVICE

    def MC_search(self, src, tgt, have_len, docs_max_length):
        tgt_input = tgt[:, :have_len]
        batch_size = tgt.size(0)

        samples = torch.zeros(batch_size, docs_max_length).long()
        samples[:, :have_len] = tgt[:, :have_len]
        samples = samples.to(self.DEVICE)
        src = torch.transpose(src, 0, 1)
        tgt = torch.transpose(tgt, 0, 1)
        tgt_input = torch.transpose(tgt_input, 0, 1)

        for i in range(have_len, docs_max_length):
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.DEVICE)
            probs = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)  # prob:batch_size*vocab_size
            # probs = torch.where(torch.isnan(probs), torch.full_like(probs, 0), probs)
            # probs = torch.log(probs)
            outs = torch.multinomial(probs[-1, :], 1)  # outs:batch_size*1
            # _, outs = torch.max(probs[-1, :], dim=1)
            samples[:, i] = outs.view(-1).data
            # samples[:, i] = outs

            # tgt_input = outs.unsqueeze(0)
            tgt_input = samples[:, :i + 1].transpose(0, 1)

        return samples

    def get_reward(self, src, tgt, rollout_num, discriminator):
        with torch.no_grad():
            batch_size = tgt.size(0)
            docs_max_length = tgt.size(1)
            rewards = torch.zeros([rollout_num * (docs_max_length - 1), batch_size]).float()  # [rollout_num * self.docs_max_length, batch_size]
            rewards = rewards.to(self.DEVICE)

            idx = 0
            for i in range(rollout_num):
                for have_len in range(2, docs_max_length + 1):
                    samples = self.MC_search(src, tgt, have_len, docs_max_length)
                    out = discriminator.forward(src, samples)  # src:batch_size*code_max_length samples:batch_size*docs_max_length
                    out = F.softmax(out, dim=-1)
                    reward = out[:, 1]  # scores/probabilities that discriminator thinks these docstrings are true
                    rewards[idx] = reward
                    idx += 1

        rewards = rewards.transpose(0, 1)  # [batch_size, rollout_num * self.docs_max_length]
        # rewards = torch.mean(rewards.view(batch_size, (docs_max_length - 1), rollout_num), dim=-1)
        rewards = torch.mean(rewards.view(batch_size, rollout_num, (docs_max_length - 1)), dim=1)  # [batch_size, self.docs_max_length]
        return rewards
