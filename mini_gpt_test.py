from mini_gpt_model import GPT
from trainer import Trainer
import torch

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from collections import Counter
import numpy as np


with open('1b_benchmark.train.tokens', 'r') as f: lines_train = f.readlines()
with open('1b_benchmark.dev.tokens', 'r') as f: lines_dev = f.readlines()
with open('1b_benchmark.test.tokens', 'r') as f: lines_test = f.readlines()

# each element is a list of tokens
tokens_train = [line.split() for line in lines_train]

print(f'train docs: {len(tokens_train)}')
print(f'total train tokens: {sum(len(t) for t in tokens_train)}')

# utility fn to flatten the tokens structure
def flat(tokens):
    for t in tokens:
        yield from t


# get counts of each token sorted by count, descending
# also add a few special tokens (with high counts) so they appear first
token_counts = Counter(flat(tokens_train))
token_counts['<START>'] = 1000004
token_counts['<STOP>'] = 1000003
token_counts['<UNK>'] = 1000002
token_counts['<PAD>'] = 1000001
sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

print('unique_tokens:', len(token_counts))
print('unique_tokens, count>=3:', len([t for t in sorted_tokens if t[1] >= 3]))

# make tokenizer for all tokens with count >= 3
# note that our tokenizer ends up including START and STOP tokens too
tokenizer = {t[0]: i for i, t in enumerate(sorted_tokens) if t[1] >= 3}

def pad_to_length(tokens, max_len, tokenizer=tokenizer):
    return tokens[:max_len] + [tokenizer['<PAD>']] * (max_len - len(tokens))

def tokenize(sentence, pad_to_len=None, include_stop=True, tokenizer=tokenizer):
    words = [tokenizer.get(w, tokenizer['<UNK>']) for w in sentence.split()]
    # add START and STOP tokens
    tokens = [tokenizer['<START>']] + words + ([tokenizer['<STOP>']] * include_stop)

    if pad_to_len is not None:
        tokens = pad_to_length(tokens, pad_to_len, tokenizer=tokenizer)
    return tokens

# invert tokenizer for decoding
tokenizer_inv = {v:k for k,v in tokenizer.items()}
def decode(tokens, tokenizer_inv=tokenizer_inv, end_at_stop=True, omit_pad=True):
    tokens = [tokenizer_inv[t] for t in tokens]
    if omit_pad:
        tokens = [t for t in tokens if t != '<PAD>']
    if end_at_stop and '<STOP>' in tokens:
        tokens = tokens[:tokens.index('<STOP>')+1]
    return ' '.join(tokens)


sentence = 'More people have said an Escher sentence than I have .'
tokenized = tokenize(sentence, pad_to_len=25) # pad to only 25 so it looks nice
decoded = decode(tokenized, end_at_stop=False, omit_pad=False)
print(f'{sentence=}\n{tokenized=}\n{decoded=}')


# Notice above that the vast majority of sequences have less than 100 tokens.
# For performance we will thus truncate to 100 tokens.

MAX_LEN = 100
DEVICE = 'cuda'

data_train = torch.tensor(
    [tokenize(t, MAX_LEN) for t in lines_train if len(t) > 0],
    dtype=torch.long
)
data_val = torch.tensor(
    [tokenize(t, MAX_LEN) for t in lines_dev if len(t) > 0],
    dtype=torch.long
)

data_train.shape, data_val.shape


# X is all but last token, Y is all but first token
train_dataset = torch.utils.data.TensorDataset(data_train[:, :-1], data_train[:, 1:])
val_dataset = torch.utils.data.TensorDataset(data_val[:, :-1], data_val[:, 1:])

# example X,Y pair from train dataset -- 2 is <START>, 3 is <STOP>
train_dataset[447]



model_config = GPT.get_default_config()
model_config.model_type = None
model_config.pad_token = tokenizer['<PAD>']


model_config.model_type = 'gpt-nano'
# 'gpt-nano' equivalent to:
# model_config.n_layer = 3
# model_config.n_head = 3
# model_config.n_embd = 48

model_config.vocab_size = max(tokenizer.values()) + 1

model_config.block_size = 1024

model_config.attn_init_fn = init_qkv_proj
model_config.attn_fn = self_attention



model = GPT(model_config)



train_config = Trainer.get_default_config()
train_config.device = DEVICE
train_config.num_workers = 2

# We didn't tune the hyperparameters at all, feel free to change
train_config.learning_rate = 5e-4
train_config.batch_size = 32
train_config.max_iters = len(train_dataset) // train_config.batch_size  # train for 1 epoch

trainer = Trainer(train_config, model, train_dataset)
log = []

model.to(DEVICE)
model.train()

bar = tqdm(total=train_config.max_iters)
@torch.no_grad()
def on_batch_end(trainer):
    log.append( trainer.loss.item() )
    bar.set_postfix(loss=trainer.loss.item())
    bar.update()

trainer.set_callback('on_batch_end', on_batch_end)
trainer.run()
bar.close()


""" EVALUATION """

sentence = 'Thank you so much Liwei and Taylor for all your help with this !'

tokens = torch.tensor([tokenize(sentence, pad_to_len=MAX_LEN)], dtype=torch.long)
X_tokens, y_tokens = tokens[:, :-1], tokens[:, 1:]

print('notice the long tail of PAD tokens: ', tokens.cpu()[0].tolist())

model.eval()
with torch.no_grad():
    logits, loss = model(X_tokens.to(DEVICE), y_tokens.to(DEVICE))
    logits, loss = logits.cpu(), loss.cpu()


also_loss = F.cross_entropy(logits.flatten(0,1), y_tokens.flatten(0,1),
                            ignore_index=tokenizer['<PAD>'])


probs = F.softmax(logits, dim=-1)

log_probs = torch.log(probs)


y_log_probs = torch.gather(log_probs, -1, y_tokens[..., None])[..., 0]

not_pad_y_log_probs = y_log_probs[y_tokens != tokenizer['<PAD>']]

# negative average of the log probs of the target tokens is exactly crossentropy loss here!
also_loss_again = -not_pad_y_log_probs.mean()

print()
print('reported loss from model:\t', loss.item())
print('manually calculated loss:\t', also_loss.item())
print('manually calculated loss again:\t', also_loss_again.item())

# we can calculate perplexity using the crossentropy loss
perplexity = torch.exp(also_loss)
print('perplexity:', perplexity.item())



"""
A utility function to calculate loss per-document for some data.
It accepts a list of strings, tokenizes, evaluates, and returns a list of floats.
"""
@torch.no_grad
def evaluate_losses(data, model=model, bs=32, progress=True, pad_to_len=MAX_LEN):
    it = range(0, len(data), bs)
    if progress: it = tqdm(it)

    out = []
    for b_start in it:
        batch = slice(b_start, b_start+bs)
        tokens = torch.tensor(
            [tokenize(t, pad_to_len=pad_to_len) for t in data[batch]],
            dtype=torch.long).to(DEVICE)
        X_tokens, y_tokens = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

        model.eval()
        logits, _ = model(X_tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        y_log_probs = torch.gather(log_probs, 2, y_tokens[..., None])[..., 0]

        for i in range(y_tokens.shape[0]):
            not_pad = (y_tokens[i] != tokenizer['<PAD>'])
            loss = -y_log_probs[i, not_pad].mean()
            out.append(loss.item())

    return out


# calculate loss and perplexity for a single sentence
is_this_loss = evaluate_losses(['After learning language models model natural language',], progress=False)[0]
print('loss:', is_this_loss)
print('perplexity:', np.exp(is_this_loss)) 


# Here's an example of generating using the model -- see generate in minGPT's model.py

sentence = ''                         # empty prompt -> sample from model at random
# sentence = 'unfortunately ,'          # can sample more negative stuff
# sentence = 'fun fact : did you know'  # AI-generated fun facts

tokens = torch.tensor([tokenize(sentence, include_stop=False)], dtype=torch.long).to(DEVICE)

for _ in range(10):
    pred = model.generate(tokens, MAX_LEN-tokens.shape[-1],
                        temperature=1.0, do_sample=True, top_k=None)

    print(decode(pred[0].tolist()))