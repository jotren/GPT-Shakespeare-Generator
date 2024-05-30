# Shakespeare GPT Model

The aim of this project is to create a GPT model that generates writings in the style of Shakespeare!

<img src="./images/shakespeare.webp" alt="Shakespeare" width=800/> 

Massive thanks to Andrej Karpathy for his lecture series on GPTs! [Watch the lecture here](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4946s).

### Data

The data used to train this model is Tiny Shakespeare: [Download the dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

In most cases, commercial GPTs tokenize data into small chunks of words. In my model, I tokenize at the letter level. This reduces the amount of data ingested by the model, resulting in about 50 tokens to consider.

```python
# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # First 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
```

These functions encode and decode the words from letters to numbers (and vice versa).

```JSON
{'\n': 0, ' ': 1, '!': 2, '$': 3, '&': 4, "'": 5, ',': 6, '-': 7, '.': 8, '3': 9, ':': 10, ';': 11, '?': 12, 'A': 13, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26, 'O': 27, 'P': 28, 'Q': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 39, 'b': 40, 'c': 41, 'd': 42, 'e': 43, 'f': 44, 'g': 45, 'h': 46, 'i': 47, 'j': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'u': 59, 'v': 60, 'w': 61, 'x': 62, 'y': 63, 'z': 64}
```

Then using PyTorch's embedding layers, each token is given a corresponding **364** vector matrix to represent the semantic space of that letter.

### Approach

o give this semantic space context, I use Multi-Head Attention transformers. The mathematics of attention is covered here: [Transformer Encoder Maths](https://github.com/jotren/Machine-Learning-Teaching-Material/blob/main/theory/Transformer%20Encoder%20Maths.md).

The transformer architecture is crucial for understanding the context of the text. The class below illustrates the code behind the mathematics:

```python
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

```
To summarise, once this transformation is completed, the embedding vectors are shifted around. The letters that occur more frequently near each other begin to move together. You can think of the resulting matrix as a dimensional representation of each letter in n-dimensional space.

This information is then fed to a neural network:

```python
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

```
Now, this is just one of many NN layers. The neural network takes a sequence of embeddings and uses complex linear transformations to understand context and predict the next items in the sequence more accurately.

Attention helps because it allows the neural network to interpret which letters (or words) are in similar semantic spaces by focusing on the most relevant parts of the input. This enables the model to make better predictions based on context.

### Testing

Then, using softmax, for each step in generation, we provide a probability that a specific letter will be selected.

```python
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
```

This allows the model to generate the text letter by letter. Below is an example output of the GPT:
```
Of most Anne, my horse and hours blank
Hath berefts rice in him: but give himself,
Though in this noble repent, or near
Mis-shelf the very time to himself; I
A rarising in't, to acquaint him mad; old
I hope sinklingly to give a crew thing.

FRIAR LAURENCE:
Ketpy for early those guilty!

ROMEO:
Sir, come still lady! What say they sad sin?

FRIAR LAURENCE:
For the deed one of his: and for what talk of justice?

ROMEO:
thou hast not, that is, thou dost sh artill.

ROMEO:
Harry, heaven mortal, for most sword to love,
That shall be at bear and majesty
Is alive, and to end so cause him in signorant,
Keep him to dare bring of.

BENVOLIO:
And, faith's forsword;
Mortal up several point, at the heart of a man.

BENVOLIO:
Stand, by flaying, take away it.
```

ou can see that the model is able to generate language, but the story is not very logical. In commercial GPTs, this process has been refined to provide language that is sensible.

### Deployment



### Key Takeaways

Transformer architecture is extremely powerful, they require extensive tuning and many layers to provide meaningful output. Project has help me understand:

- The intricacies of self-attention
- How to tokenize from scratch.
- How to Tune hyperparameters to provide a decent sensible output.
