# -*- coding: utf-8 -*-
"""DL_HW3_2022_text_generation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LmPkuBaWGBWG9eX50L6iBJWhg5j-gsVZ

# Homework 3 - Text generation with LSTM and Transformer networks

## Installs the unidecode library and downloads the Shakespeare dataset.
"""

#!pip install unidecode
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

"""## LSTM implementation

For this task you will implement the LSTM neural network architecture and train it on the task of character-level text generation. Implement a single layer LSTM and optionally extend your implementation to multiple layers to generate better results.

Links:

- https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html -- Lists the equations for each component of the LSTM cell.
- http://colah.github.io/posts/2015-08-Understanding-LSTMs/ -- Intuitive explanation of LSTM
- http://karpathy.github.io/2015/05/21/rnn-effectiveness/ -- Explanation and uses of RNNs.


Implement the initialization and the forward pass of a LSTMCell and use it as part of the LSTMSimple network class. 

The input of the LSTM network will be a sequence of characters, whereas the input of the LSTMCell will be a single input character (x), the output of the previous iteration (C) and the hidden state of the previous iteration (h). Iteratively process the entire input character sequence and calculate the loss based on the prediction at each time step. 

### Do NOT use the torch.nn.LSTM class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
torch.set_printoptions(profile="full")

class LSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super(LSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        ## Initialize the necessary components
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Initialize weights and biases
        self.ii = nn.Linear(input_dim, hidden_dim)
        self.hi = nn.Linear(hidden_dim, hidden_dim)
        self.iif = nn.Linear(input_dim, hidden_dim)
        self.hf = nn.Linear(hidden_dim, hidden_dim)
        self.ig = nn.Linear(input_dim, hidden_dim)
        self.hg = nn.Linear(hidden_dim, hidden_dim)
        self.io = nn.Linear(input_dim, hidden_dim)
        self.ho = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, C, h):
        '''
        Input:
        x - batch of encoded characters
        C - Cell state of the previous iteration
        h - Hidden state of the previous iteration
        Output: 
        c_out - cell state 
        h_out - hidden state
        '''
        i = self.sigmoid(self.ii(x) + self.hi(h))
        f = self.sigmoid(self.iif(x) + self.hf(h))
        g = self.tanh(self.ig(x) + self.hg(h))
        o = self.sigmoid(self.io(x) + self.ho(h))
        c_out = f * C + i * g 
        h_out = o*self.tanh(c_out)

        return c_out, h_out

class LSTMSimple(nn.Module):
    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTMSimple, self).__init__()

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        print("seq_length: ", seq_length)
        print("input_dim: ", input_dim)
        print("hidden_dim: ", hidden_dim)
        print("output_dim: ", output_dim)
        print("batch_size: ", batch_size)

        self.proj = nn.Linear(hidden_dim, output_dim)
        self.lstm_cell = LSTMCell(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        # x - One hot encoded batch - Shape: (batch, seq_len, onehot_char)
        # Returns the predicted next character for each character in the 
        # sequence
        # Implement the forward pass over the sequence of characters
        self.c = torch.zeros(x.size(0), self.hidden_dim).to('cuda')
        self.h = torch.zeros(x.size(0), self.hidden_dim).to('cuda')

        # batch - number of sequences until the weight update (optimizer step)
        # seq_len = chunk_len = chunk_size # number of items to learn the sequence from to predict the output - seq_len is sentence of seq_len chars 
        # onehot_char is a array of len(all_characters), where each character has a unique array with only one 1
        #print("batch: ", x.shape) # [256, 128, 100]
        self.seq_length = x.size(1)
        #print(self.seq_length)

        out = torch.zeros(x.size(0), self.seq_length, self.hidden_dim).to('cuda')
        #print("OUT SHAPE: ", out.shape)
        # Iteratively process the entire input character sequence
        for j in range(self.seq_length):
            self.c, self.h = self.lstm_cell(x[:, j, :], self.c, self.h)
            #print(self.h.shape)
            out[:, j, :] = self.h
 
        # output shape more bit [256, 128]
        # za vsako črko v 128-črkovnem besedili v batchu 1 predikcija
        # torej skupaj 128 predikcij na eno besedilo
        # oz. 256*128 predikcij za 256 batchov
        # mreža pa se uči iz onehot encodingov
        out = self.proj(out)

        return out, (self.h, self.c)

### LSTM Sampling Code
"""

To generate text the network must predict the next character in a sequence, however networks do not produce a single character but rather estimate the likelihood for each possible character. Sampling characters from the network output can be done in different ways with common ones being the Greedy sampling process and Top-K sampling.

In the simple greedy sampling method the network takes a text prompt as input and generates an additional N tokens by always taking the token with the highest prediction score as the next token.

In the Top-K sampling, randomness is added to the sampling process as the network samples from K most likely predicitons at each step. This alleviates the problem of generative models repeating text but may generate incorrect text by sampling inappropriate tokens.
"""

def greedy_sampling_lstm(lstm, x, num_chars):
    # x -- b x onehot_char
    outputs = torch.zeros((1, num_chars, x.shape[2]))
    t_outputs, (cell_state, hidden) = lstm(x.float())
    for c in range(num_chars):
        output_tmp = torch.softmax(lstm.proj(hidden),dim=1)
        top_ind = torch.argmax(output_tmp, dim=1)[0]
        tmp = torch.zeros_like(x[:, 0, :]).cuda()
        tmp[:,top_ind] = 1
        outputs[:,c] = tmp

        cell_state, hidden = lstm.lstm_cell(tmp,cell_state,hidden)
    return outputs

def topk_sampling_lstm(lstm, x, num_chars):
    # x -- b x onehot_char
    outputs = torch.zeros((1,num_chars,x.shape[2]))
    t_outputs, (cell_state, hidden) = lstm(x.float())
    for c in range(num_chars):
        output_vals, output_ind = torch.topk(lstm.proj(hidden), 5, dim=1)
        output_tmp = torch.softmax(output_vals,dim=1)
        top_ind = torch.multinomial(output_tmp[0], 1)[0]
        tmp = torch.zeros_like(x[:,0,:]).cuda()
        tmp[:,output_ind[0,top_ind]] = 1
        outputs[:,c] = tmp

        cell_state, hidden = lstm.lstm_cell(tmp,cell_state,hidden)

    return outputs

"""### LSTM Dataset Code"""

import unidecode
import string
import random
from torch.autograd import Variable
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    def __init__(self, chunk_len=200, padded_chunks=False):
        # Character based dataset
        dataset_path = "./input.txt"
        # The tokens in the vocabulary (all_characters)
        # are just the printable characters of the string class
        self.all_characters = string.printable # abcde..ABCD..-/{..
        self.n_characters = len(self.all_characters)
        print(self.n_characters)
        # Maps characters to indices
        self.char_dict = {x:i for i,x in enumerate(self.all_characters)}
        self.file, self.file_len = self.read_file(dataset_path)
        # Sequence length of the input
        self.chunk_len = chunk_len

    def read_file(self, filename):
        file = unidecode.unidecode(open(filename).read())
        return file, len(file)
    
    def char_tensor(self, in_str):
        # in_str - input sequence - String
        # Return one-hot encoded characters of in_str
        tensor = torch.zeros(len(in_str),self.n_characters).long()
        char_ind = [self.char_dict[c] for c in in_str]
        tensor[torch.arange(tensor.shape[0]),char_ind] = 1
        return tensor

    def __getitem__(self, idx):
        inp, target = self.get_random_text()
        return {"input":inp, "target":target}

    def __len__(self):
        return 10000

    def get_random_text(self):
        # Pick a random string of length self.chunk_len from the dataset
        start_index = np.random.randint(0, self.file_len - self.chunk_len)
        end_index = start_index + self.chunk_len + 1
        chunk = self.file[start_index:end_index]
        # One-hot encode the chosen string
        inp = self.char_tensor(chunk[:-1])
        # The target string is the same as the
        # input string but shifted by 1 character
        target = self.char_tensor(chunk[1:])
        inp = Variable(inp).cuda()
        target = Variable(target).cuda()
        return inp, target

"""### LSTM Training loop

With a correct implementation you should get sensible text generation results with the set parameters, however you should experiment with various parameters,
especially with the sequence length (chunk_len) used during training.
"""

from tqdm import tqdm
import torch.optim as optim

batch_size = 256
chunk_len = 128
model_name = "LSTM"
train_dataset = LSTMDataset(chunk_len=chunk_len)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)

#Sample parameters, use whatever you see fit.
input_dim = train_dataset.n_characters
hidden_dim = 256
output_dim = train_dataset.n_characters
learning_rate = 0.005
model = LSTMSimple(chunk_len,input_dim, hidden_dim, output_dim,batch_size)
model.train()
model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs=30

for epoch in range(epochs):
    with tqdm(total=len(trainloader.dataset), desc ='Training - Epoch: '+str(epoch)+"/"+str(epochs), unit='chunks') as prog_bar:
        for i, data in enumerate(trainloader, 0):
            inputs = data['input'].float()
            labels = data['target'].float()
            # b x chunk_len x len(dataset.all_characters)
            target = torch.argmax(labels,dim=2)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(inputs.shape[0]*inputs.shape[1],-1),target.view(labels.shape[0]*labels.shape[1]))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                      max_norm=10.0)
            optimizer.step()
            prog_bar.set_postfix(**{'run:': model_name,'lr': learning_rate,
                                    'loss': loss.item()
                                    })
            prog_bar.update(batch_size)
        # Intermediate output
        sample_text = "O Romeo, wherefore art thou"
        inp = train_dataset.char_tensor(sample_text)
        sample_input = Variable(inp).cuda().unsqueeze(0).float()
        out_test = topk_sampling_lstm(model,sample_input, 300)[0]
        out_char_index = torch.argmax(out_test, dim=1).detach().cpu().numpy()
        out_chars = sample_text+"".join([train_dataset.all_characters[i] for i in out_char_index])
        print("Top-K sampling -----------------")
        print(out_chars)

        out_test = greedy_sampling_lstm(model,sample_input, 300)[0]
        out_char_index = torch.argmax(out_test, dim=1).detach().cpu().numpy()
        out_chars = sample_text+"".join([train_dataset.all_characters[i] for i in out_char_index])
        print("Greedy sampling ----------------")
        print(out_chars)
