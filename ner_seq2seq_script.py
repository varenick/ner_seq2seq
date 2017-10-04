import numpy as np

raw_data = {}
input_vocab = set()
output_vocab = set()
modes = ['train', 'valid', 'test']

for mode in modes:
    raw_data[mode] = [[]]
    with open(mode + '.txt') as f:
        for line in f:
            line_split = line.split()
            if len(line_split) != 4:
                if raw_data[mode][-1]:
                    raw_data[mode].append([])
                continue
            word, _, _, tag = line_split
            word = word.lower()
            raw_data[mode][-1].append((word, tag))
            input_vocab.add(word)
            output_vocab.add(tag)

PAD_id = 0

id_to_word = [''] + list(input_vocab)
word_to_id = {word: i for i, word in enumerate(id_to_word)}

id_to_tag = [''] + list(output_vocab)
tag_to_id = {tag: i for i, tag in enumerate(id_to_tag)}

input_vocab_size = len(id_to_word)
output_vocab_size = len(id_to_tag)

MAX_SAMPLE_LENGTH = 20

word_ids = {}
tag_ids = {}

for mode in modes:
    word_ids[mode] = np.zeros((len(raw_data[mode][:-1]), MAX_SAMPLE_LENGTH), dtype=np.long) + PAD_id
    tag_ids[mode] = np.zeros((len(raw_data[mode][:-1]), MAX_SAMPLE_LENGTH), dtype=np.long) + PAD_id
    for i, sample in enumerate(raw_data[mode][:-1]):
        for j, (word, tag) in enumerate(sample):
            if j >= MAX_SAMPLE_LENGTH:
                break
            word_ids[mode][i][j] = word_to_id[word]
            tag_ids[mode][i][j] = tag_to_id[tag]
            
def get_batch(word_ids, tag_ids, batch_size):
    samples_count = word_ids.shape[0]
    while True:
        batch_samples_idx = np.random.choice(samples_count, size=(batch_size,))
        yield word_ids[batch_samples_idx], tag_ids[batch_samples_idx]
        
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
    
from seq2seq_model import Seq2SeqModel

loss_function = nn.CrossEntropyLoss()

import os

def f1_score_script(sample_batch, tag_batch, outputs_np,
                    test_output_fname='test_output.txt', report_fname='report.txt',
                    print_result=True):
    test_batch_size = sample_batch.shape[0]
    with open(test_output_fname, 'w') as f:
        for i in range(test_batch_size):
            for j in range(MAX_SAMPLE_LENGTH):
                word_id = sample_batch[i][j]
                tag_true_id = tag_batch[i][j]
                tag_pred_id = outputs_np[i][j]
                if word_id == 0:
                    break
                word = id_to_word[word_id]
                tag_true = id_to_tag[tag_true_id]
                tag_pred = id_to_tag[tag_pred_id]
                if tag_pred_id == 0:
                    tag_pred = 'O'
                f.write(' '.join([word] + ['pur'] * 4 + [tag_true] + [tag_pred]) + '\n')
            f.write('\n')
            
    conll_evaluation_script = os.path.join('.', 'conlleval')
    shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, test_output_fname, report_fname)

    score = 0
    os.system(shell_command)
    with open(report_fname) as f:
        for i, line in enumerate(f):
            if i == 1:
                score = float(line.split()[-1])
            if print_result:
                print(line)
            elif i == 1:
                break
    return score
    
from time import time

train_batch_size = 64
eval_batch_size = 1024
decode_batch_size = 4
test_batch_size = 2048

train_batch_gen = get_batch(word_ids['train'], tag_ids['train'], batch_size=train_batch_size)
eval_batch_gen = get_batch(word_ids['valid'], tag_ids['valid'], batch_size=eval_batch_size)
test_batch_gen = get_batch(word_ids['test'], tag_ids['test'], batch_size=test_batch_size)

model_params = {
    'input_vocab_size': input_vocab_size,
    'output_vocab_size': output_vocab_size,
    'word_embed_dim': 300,
    'tag_embed_dim': 8,
    'num_hidden': 64
}

num_runs = 10

num_steps = 3000
print_skip = 100
eval_skip = 10

train_losses = []

eval_losses = []
eval_losses_noisy = []

eval_f1s = []
eval_f1s_noisy = []

output_mode = 'argmax'
feed_mode = 'soft-gumbel'
reinforce_strategy = 'none'
if feed_mode != 'sampling':
    reinforce_strategy = ''
softmax_temperature = 1.
if feed_mode not in ['soft-gumbel', 'hard-gumbel']:
    softmax_temperature = ''
    
attention_mode = 'fixed'

mode_name = feed_mode + '_' + str(softmax_temperature) + '_' + reinforce_strategy + '_' + attention_mode

do_eval = True

av_advantage = None
std_advantage = None

grad_norms = None

print(mode_name)

for run in range(num_runs):
    print('Run', run)
    print()
    
    train_losses.append([])
    cum_train_loss = 0

    train_av_loss = 0
    batch_av_train_av_loss = 0

    eval_losses.append([])
    eval_losses_noisy.append([])
    
    eval_f1s.append([])
    eval_f1s_noisy.append([])
    
    global_start_time = time()
    last_print_time = global_start_time

    model = Seq2SeqModel(**model_params).cuda()
    
    av_advantage = []
    std_advantage = []
    
    grad_norms = []

    init_lr = 0.001
    lr = init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for step in range(num_steps):        
        # Train:
        sample_batch, tag_batch = next(train_batch_gen)
    
        sample_batch_torch = autograd.Variable(torch.from_numpy(sample_batch).cuda(), requires_grad=False)
        tag_batch_torch = autograd.Variable(torch.from_numpy(tag_batch).cuda(), requires_grad=False)
    
        if reinforce_strategy == 'argmax_advantage':
            unscaled_logits, unscaled_logits_baseline, outputs = model(
                sample_batch_torch, tag_batch_torch,
                output_mode=output_mode, feed_mode=feed_mode, baseline_mode='argmax',
                attention_mode=attention_mode, dropout_rate=0.2
            )
        else:
            unscaled_logits, _, outputs = model(
                sample_batch_torch, tag_batch_torch,
                output_mode=output_mode, feed_mode=feed_mode, baseline_mode=None,
                attention_mode=attention_mode, dropout_rate=0.2, softmax_temperature=softmax_temperature
            )
        train_loss = loss_function(unscaled_logits.view(-1, output_vocab_size), tag_batch_torch.view(-1))
        
        if reinforce_strategy == 'element':
            tag_batch_torch_one_hot = torch.zeros(train_batch_size, chunk_length, output_vocab_size).cuda()
            tag_batch_torch_one_hot.scatter_(
                2, tag_batch_torch.data.view(train_batch_size, chunk_length, 1), 1
            )
            elemwise_train_loss = (-1) * F.log_softmax(
                unscaled_logits.data.view(-1, vocab_size)
            ).data.view(train_batch_size, chunk_length, vocab_size)[rev_chunk_batch_torch_one_hot.byte()].view(
                train_batch_size, chunk_length
            )
            batch_av_train_loss = torch.mean(elemwise_train_loss, dim=0)
            if step == 0:
                batch_av_train_av_loss = batch_av_train_loss
            else:
                batch_av_train_av_loss = 0.99 * batch_av_train_av_loss + 0.01 * batch_av_train_loss
            normed_batch_centered_train_loss = ((elemwise_train_loss - batch_av_train_av_loss) / 
                                                (train_batch_size * chunk_length))
            seqwise_train_loss = torch.sum(normed_batch_centered_train_loss, dim=1)
            seqwise_cum_train_loss = torch.cumsum(normed_batch_centered_train_loss, dim=1)
            for t, dec_feed in enumerate(model.dec_feeds):
                dec_feed.reinforce(
                    (-1) * (seqwise_train_loss - seqwise_cum_train_loss[:, t]).view(train_batch_size, 1)
                )
                
        elif reinforce_strategy == 'argmax_advantage':
            tag_batch_torch_one_hot = torch.zeros(train_batch_size, MAX_SAMPLE_LENGTH, output_vocab_size).cuda()
            tag_batch_torch_one_hot.scatter_(
                2, tag_batch_torch.data.view(train_batch_size, MAX_SAMPLE_LENGTH, 1), 1
            )
            elemwise_train_loss = (-1) * F.log_softmax(
                unscaled_logits.data.view(-1, output_vocab_size)
            ).data.view(train_batch_size, MAX_SAMPLE_LENGTH, output_vocab_size)[tag_batch_torch_one_hot.byte()].view(
                train_batch_size, MAX_SAMPLE_LENGTH
            )
            elemwise_train_loss_baseline = (-1) * F.log_softmax(
                unscaled_logits_baseline.data.view(-1, output_vocab_size)
            ).data.view(train_batch_size, MAX_SAMPLE_LENGTH, output_vocab_size)[tag_batch_torch_one_hot.byte()].view(
                train_batch_size, MAX_SAMPLE_LENGTH
            )
            normed_elemwise_advantage = ((elemwise_train_loss_baseline - elemwise_train_loss) /
                                         (train_batch_size * MAX_SAMPLE_LENGTH))
            sum_normed_elemwise_advantage = torch.sum(normed_elemwise_advantage, dim=1)
            cumsum_normed_elemwise_advantage = torch.cumsum(normed_elemwise_advantage, dim=1)
            for t, dec_feed in enumerate(model.dec_feeds):
                dec_feed.reinforce(
                    (sum_normed_elemwise_advantage - cumsum_normed_elemwise_advantage[:, t]).view(train_batch_size, 1)
                )
                
            av_advantage.append(torch.mean(elemwise_train_loss_baseline - elemwise_train_loss, dim=0).cpu().numpy())
            std_advantage.append(torch.std(elemwise_train_loss_baseline - elemwise_train_loss, dim=0).cpu().numpy())
            
        elif reinforce_strategy == 'sequence':
            rev_chunk_batch_torch_one_hot = torch.zeros(train_batch_size, chunk_length, vocab_size).cuda()
            rev_chunk_batch_torch_one_hot.scatter_(
                2, rev_chunk_batch_torch.data.view(train_batch_size, chunk_length, 1), 1
            )
            elemwise_train_loss = (-1) * F.log_softmax(
                unscaled_logits.data.view(-1, vocab_size)
            ).data.view(train_batch_size, chunk_length, vocab_size)[rev_chunk_batch_torch_one_hot.byte()].view(
                train_batch_size, chunk_length
            )
            if step == 0:
                train_av_loss = train_loss.data
            else:
                train_av_loss = 0.99 * train_av_loss + 0.01 * train_loss.data
            seq_av_train_loss = torch.mean(elemwise_train_loss, dim=1, keepdim=True)
            for t, dec_feed in enumerate(model.dec_feeds):
                dec_feed.reinforce(
                    (-1) * (seq_av_train_loss - train_av_loss) / (train_batch_size * chunk_length)
                )
                
        elif reinforce_strategy == 'none':
            for t, dec_feed in enumerate(model.dec_feeds):
                dec_feed.reinforce(torch.zeros(train_batch_size, 1).cuda())
        elif reinforce_strategy:
            raise ValueError("Invalid reinforce_strategy: '{}'".format(reinforce_strategy))
            
        optimizer.zero_grad()
        train_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=4)
        #lr = init_lr / (step + 1) ** 0.5
        #for group in optimizer.param_groups:
        #    group['lr'] = lr
        optimizer.step()
        
        grad_norms.append([])
        for param in model.parameters():
            grad_norms[-1].append(torch.norm(param.grad.data))
        
        train_losses[-1].append(train_loss.data.cpu().numpy().mean())

        cum_train_loss += train_losses[-1][-1]
        
        # Eval:
        if do_eval and (step + 1) % eval_skip == 0:
            sample_batch, tag_batch = next(eval_batch_gen)
    
            sample_batch_torch = autograd.Variable(torch.from_numpy(sample_batch).cuda(), requires_grad=False)
            tag_batch_torch = autograd.Variable(torch.from_numpy(tag_batch).cuda(), requires_grad=False)
    
            unscaled_logits, _, outputs = model(
                sample_batch_torch,
                output_mode='argmax', feed_mode='same', attention_mode=attention_mode, dropout_rate=0.0
            )
            eval_loss = loss_function(unscaled_logits.view(-1, output_vocab_size), tag_batch_torch.view(-1))
            eval_f1 = f1_score_script(
                sample_batch, tag_batch, outputs.data.cpu().numpy(),
                print_result=False,
                test_output_fname='test_output_' + mode_name + '.txt',
                report_fname='report_' + mode_name + '.txt'
            )
            
            eval_losses[-1].append(eval_loss.data.cpu().numpy().mean())
            eval_f1s[-1].append(eval_f1)
        
            unscaled_logits, _, outputs = model(
                sample_batch_torch,
                output_mode='argmax', feed_mode='noise', attention_mode=attention_mode, dropout_rate=0.0
            )
            eval_loss_noisy = loss_function(unscaled_logits.view(-1, output_vocab_size), tag_batch_torch.view(-1))
            eval_f1_noisy = f1_score_script(
                sample_batch, tag_batch, outputs.data.cpu().numpy(),
                print_result=False,
                test_output_fname='test_output_' + mode_name + '.txt',
                report_fname='report_' + mode_name + '.txt'
            )
            
            eval_losses_noisy[-1].append(eval_loss_noisy.data.cpu().numpy().mean())
            eval_f1s_noisy[-1].append(eval_f1_noisy)
        
            # Print:
        if (step + 1) % print_skip == 0:
            print('Step', step + 1)
            
            print('Train loss: {:.2f}'.format(
                cum_train_loss / print_skip,
            ))
            cum_train_loss = 0
            
            if do_eval:
                print('Eval loss: {:.2f}; same, with noise on input: {:.2f}'.format(
                    eval_losses[-1][-1], eval_losses_noisy[-1][-1]
                ))
                print('Eval F1: {:.2f}; same, with noise on input: {:.2f}'.format(
                    eval_f1, eval_f1_noisy
                ))
            
            print('{:.2f}s from last print'.format(time() - last_print_time))
            last_print_time = time()
            print()
    
    print('{} steps took {:.2f}s\n'.format(num_steps, time() - global_start_time))
    
train_losses_mean = np.mean(train_losses, axis=0)
train_losses_std = np.std(train_losses, axis=0)

eval_losses_mean = np.mean(eval_losses, axis=0)
eval_losses_std = np.std(eval_losses, axis=0)

eval_losses_noisy_mean = np.mean(eval_losses_noisy, axis=0)
eval_losses_noisy_std = np.std(eval_losses_noisy, axis=0)

eval_f1s_mean = np.mean(eval_f1s, axis=0)
eval_f1s_std = np.std(eval_f1s, axis=0)

eval_f1s_noisy_mean = np.mean(eval_f1s_noisy, axis=0)
eval_f1s_noisy_std = np.std(eval_f1s_noisy, axis=0)

import pickle

if not os.path.exists('./' + mode_name):
    os.makedirs('./' + mode_name)
for name in ['train_losses', 'eval_losses', 'eval_f1s']:
    for infix in ['', '_noisy']:
        if name == 'train_losses' and infix == '_noisy':
            continue
        for suffix in ['_mean', '_std', '']:
            with open(mode_name + '/' + name + infix + suffix + '.dat', 'wb') as f:
                pickle.dump(eval(name + infix + suffix), f)
