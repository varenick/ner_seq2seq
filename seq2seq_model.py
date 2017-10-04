import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

def sample_gumbel(input):
    noise = torch.rand(input.size()).cuda()
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return autograd.Variable(noise)

def gumbel_softmax_sample(input, temperature, hard=False):
    noise = sample_gumbel(input)
    x = (input + noise) / temperature
    x = F.softmax(x)

    if hard:
        max_val, _ = torch.max(x, x.dim()-1)
        x_hard = x == max_val.unsqueeze(-1).expand_as(x)
        tmp = (x_hard.float() - x)
        tmp2 = tmp.clone()
        tmp2.detach_()
        x = tmp2 + x

    return x.view_as(input)
    
class Seq2SeqModel(nn.Module):
    def __init__(
        self, input_vocab_size, output_vocab_size, word_embed_dim, tag_embed_dim,
        num_hidden, num_layers=1
    ):
        super(Seq2SeqModel, self).__init__()
        
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.word_embed_dim = word_embed_dim
        self.tag_embed_dim = tag_embed_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        
        self.input_embedding = nn.Embedding(input_vocab_size, word_embed_dim)
        self.enc_rnn = nn.LSTM(word_embed_dim, num_hidden, num_layers,
                               batch_first=True, bidirectional=True)
        self.feed_embedding = nn.Embedding(output_vocab_size, tag_embed_dim)
        self.dec_cell = nn.LSTMCell(tag_embed_dim, num_hidden * num_layers * 2)
        self.output_proj = nn.Linear(num_hidden * num_layers * 2, output_vocab_size)
        self.attn_combine = nn.Linear(num_hidden * 4, num_hidden * 2)
        
    def forward(
        self, input_chunk, target_chunk=None, feed_mode='same', output_mode='argmax', baseline_mode=None,
        softmax_temperature=1., attention_mode=None, dropout_rate=0.2
    ):
        need_target_chunk = (feed_mode == 'teacher-forcing')
        if target_chunk is None and need_target_chunk:
            raise ValueError("You should provide target_chunk when using feed_mode '{}'".format(feed_mode))
        if feed_mode not in ['same', 'teacher-forcing', 'argmax', 'sampling', 'soft-gumbel', 'hard-gumbel', 'noise']:
            raise ValueError("Invalid feed_mode: '{}'".format(feed_mode))
        if output_mode not in ['argmax', 'sampling']:
            raise ValueError("Invalid output_mode: '{}'".format(output_mode))
        if baseline_mode not in [None, 'argmax']:
            raise ValueError("Invalid baseline_mode: '{}'".format(baseline_mode))
        if attention_mode not in [None, 'fixed']:
            raise ValueError("Invalid attention_mode: '{}'".format(attention_mode))
            
        batch_size, chunk_length = input_chunk.size()
        
        # Encoder:
        input_chunk_emb = self.input_embedding(input_chunk)
        enc_h_first = autograd.Variable(
            torch.zeros(self.num_layers * 2, batch_size, self.num_hidden).cuda(),
            requires_grad=False
        )
        enc_c_first = autograd.Variable(
            torch.zeros(self.num_layers * 2, batch_size, self.num_hidden).cuda(),
            requires_grad=False
        )
        enc_hs, enc_hc_last = self.enc_rnn(input_chunk_emb, (enc_h_first, enc_c_first))
        
        attn_applied = enc_hs
        if attention_mode == 'fixed':
            attn_applied = enc_hs
        elif attention_mode is not None:
            raise ValueError("Invalid attention_mode: '{}'".format(attention_mode))
        
        # Decoder:
        self.dropout = nn.Dropout(p=dropout_rate)

        dec_h = torch.transpose(enc_hc_last[0], 0, 1).contiguous().view(batch_size, -1)
        dec_c = torch.transpose(enc_hc_last[1], 0, 1).contiguous().view(batch_size, -1)
        
        dec_h_baseline = torch.transpose(enc_hc_last[0], 0, 1).contiguous().view(batch_size, -1)
        dec_c_baseline = torch.transpose(enc_hc_last[1], 0, 1).contiguous().view(batch_size, -1)
        
        dec_feed = None
        dec_feed_emb = autograd.Variable(
            torch.zeros(batch_size, self.tag_embed_dim).cuda(),
            requires_grad=False
        )
        
        dec_feed_baseline = None
        dec_feed_baseline_emb = autograd.Variable(
            torch.zeros(batch_size, self.tag_embed_dim).cuda(),
            requires_grad=False
        )
        
        dec_unscaled_logits = []
        dec_unscaled_logits_baseline = []
        dec_outputs = []
        dec_outputs_baseline = []
        self.dec_feeds = []
        
        target_chunk_emb = None
        if need_target_chunk:
            target_chunk_emb = self.feed_embedding(target_chunk)
            
        for t in range(chunk_length):
            if attention_mode == 'fixed':
                dec_h = F.relu(self.attn_combine(torch.cat((dec_h, attn_applied[:, t]), 1)))
            dec_h, dec_c = self.dec_cell(self.dropout(dec_feed_emb), (dec_h, dec_c))
            dec_unscaled_logits.append(self.output_proj(dec_h))
            
            if baseline_mode is not None:
                if attention_mode == 'fixed':
                    dec_h_baseline = F.relu(self.attn_combine(torch.cat((dec_h_baseline, attn_applied[:, t]), 1)))
                dec_h_baseline, dec_c_baseline = self.dec_cell(
                    dec_feed_baseline_emb, (dec_h_baseline, dec_c_baseline)
                )
                dec_unscaled_logits_baseline.append(self.output_proj(dec_h_baseline))
            
            if output_mode == 'argmax':
                dec_outputs.append(torch.max(dec_unscaled_logits[-1], dim=1)[1])
                if baseline_mode is not None:
                    dec_outputs_baseline.append(torch.max(dec_unscaled_logits_baseline[-1], dim=1)[1])
            elif output_mode == 'sampling':
                dec_outputs.append(torch.multinomial(torch.exp(dec_unscaled_logits[-1]), 1).view(batch_size))
                if baseline_mode is not None:
                    dec_outputs_baseline.append(torch.multinomial(torch.exp(dec_unscaled_logits_baseline[-1]), 1).view(batch_size))
            else:
                raise ValueError("Invalid output_mode: '{}'".format(output_mode))
                
            if feed_mode == 'same':
                dec_feed = dec_outputs[-1]
                dec_feed_emb = self.feed_embedding(dec_feed.view(batch_size, 1)).view(batch_size, self.tag_embed_dim)
            elif feed_mode == 'teacher-forcing':
                dec_feed = target_chunk[:, t]
                dec_feed_emb = target_chunk_emb[:, t]
            elif feed_mode == 'argmax':
                dec_feed = torch.max(dec_unscaled_logits[-1], dim=1)[1]
                dec_feed_emb = self.feed_embedding(dec_feed.view(batch_size, 1)).view(batch_size, self.tag_embed_dim)
            elif feed_mode == 'sampling':
                dec_feed = torch.multinomial(F.softmax(dec_unscaled_logits[-1]), 1)
                dec_feed_emb = self.feed_embedding(dec_feed.view(batch_size, 1)).view(batch_size, self.tag_embed_dim)
            elif feed_mode == 'hard-gumbel':
                dec_feed_distr = gumbel_softmax_sample(
                    F.softmax(dec_unscaled_logits[-1]), softmax_temperature, hard=True
                )
                dec_feed = (
                    torch.matmul(dec_feed_distr, autograd.Variable(torch.arange(0, self.output_vocab_size).cuda()))
                ).long()
                #dec_feed_emb = self.feed_embedding(dec_feed.view(batch_size, 1)).view(batch_size, self.tag_embed_dim)
                dec_feed_emb = torch.matmul(dec_feed_distr, self.feed_embedding.weight)
            elif feed_mode == 'soft-gumbel':
                dec_feed_distr = gumbel_softmax_sample(
                    F.softmax(dec_unscaled_logits[-1]), softmax_temperature, hard=False
                )
                dec_feed = (
                    torch.matmul(dec_feed_distr, autograd.Variable(torch.arange(0, self.output_vocab_size).cuda()))
                ).long()
                dec_feed_emb = torch.matmul(dec_feed_distr, self.feed_embedding.weight)
            elif feed_mode == 'noise':
                dec_feed = torch.multinomial(
                    autograd.Variable(
                        torch.ones(batch_size, self.output_vocab_size).cuda(), requires_grad=False
                    ), 1
                )
                dec_feed_emb = self.feed_embedding(dec_feed.view(batch_size, 1)).view(batch_size, self.tag_embed_dim)
            else:
                raise ValueError("Invalid feed_mode: '{}'".format(feed_mode))
            self.dec_feeds.append(dec_feed)
        
            if baseline_mode == 'argmax':
                dec_feed_baseline = torch.max(dec_unscaled_logits_baseline[-1], dim=1)[1]
                dec_feed_baseline_emb = self.feed_embedding(
                    dec_feed_baseline.view(batch_size, 1)
                ).view(batch_size, self.tag_embed_dim)
            elif baseline_mode is not None:
                raise ValueError("Invalid baseline_mode: '{}'".format(baseline_mode))
        
        if baseline_mode is not None:
            return (
                torch.stack(dec_unscaled_logits, dim=1), 
                torch.stack(dec_unscaled_logits_baseline, dim=1), 
                torch.stack(dec_outputs, dim=1),
                torch.stack(dec_outputs_baseline, dim=1)
            )
        else:
            return (
                torch.stack(dec_unscaled_logits, dim=1), 
                None,
                torch.stack(dec_outputs, dim=1),
                None
            )
