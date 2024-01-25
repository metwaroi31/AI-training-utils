from __future__ import unicode_literals, print_function, division
import random
from constants import DEVICE
from data_core.data_loader import (
    prepareData,
    get_dataloader
)
from models.attentions import AttnDecoderRNN
from models.encoder import EncoderRNN
from tools.train import train

# SOS_token = 0
# EOS_token = 1


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, dropout_p=0.1):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size

#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
#         self.dropout = nn.Dropout(dropout_p)

#     def forward(self, input):
#         embedded = self.dropout(self.embedding(input))
#         output, hidden = self.gru(embedded)
#         return output, hidden
# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
#         self.out = nn.Linear(hidden_size, output_size)

#     def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
#         batch_size = encoder_outputs.size(0)
#         decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
#         decoder_hidden = encoder_hidden
#         decoder_outputs = []

#         for i in range(MAX_LENGTH):
#             decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
#             decoder_outputs.append(decoder_output)

#             if target_tensor is not None:
#                 # Teacher forcing: Feed the target as the next input
#                 decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
#             else:
#                 # Without teacher forcing: use its own predictions as the next input
#                 _, topi = decoder_output.topk(1)
#                 decoder_input = topi.squeeze(-1).detach()  # detach from history as input

#         decoder_outputs = torch.cat(decoder_outputs, dim=1)
#         decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
#         return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

#     def forward_step(self, input, hidden):
#         output = self.embedding(input)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.out(output)
#         return output, hidden


# class BahdanauAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(BahdanauAttention, self).__init__()
#         self.Wa = nn.Linear(hidden_size, hidden_size)
#         self.Ua = nn.Linear(hidden_size, hidden_size)
#         self.Va = nn.Linear(hidden_size, 1)

#     def forward(self, query, keys):
#         scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
#         scores = scores.squeeze(2).unsqueeze(1)

#         weights = F.softmax(scores, dim=-1)
#         context = torch.bmm(weights, keys)

#         return context, weights

# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1):
#         super(AttnDecoderRNN, self).__init__()
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.attention = BahdanauAttention(hidden_size)
#         self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(dropout_p)

#     def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
#         batch_size = encoder_outputs.size(0)
#         decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
#         decoder_hidden = encoder_hidden
#         decoder_outputs = []
#         attentions = []

#         for i in range(MAX_LENGTH):
#             decoder_output, decoder_hidden, attn_weights = self.forward_step(
#                 decoder_input, decoder_hidden, encoder_outputs
#             )
#             decoder_outputs.append(decoder_output)
#             attentions.append(attn_weights)

#             if target_tensor is not None:
#                 # Teacher forcing: Feed the target as the next input
#                 decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
#             else:
#                 # Without teacher forcing: use its own predictions as the next input
#                 _, topi = decoder_output.topk(1)
#                 decoder_input = topi.squeeze(-1).detach()  # detach from history as input

#         decoder_outputs = torch.cat(decoder_outputs, dim=1)
#         decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
#         attentions = torch.cat(attentions, dim=1)

#         return decoder_outputs, decoder_hidden, attentions


#     def forward_step(self, input, hidden, encoder_outputs):
#         embedded =  self.dropout(self.embedding(input))

#         query = hidden.permute(1, 0, 2)
#         context, attn_weights = self.attention(query, encoder_outputs)
#         input_gru = torch.cat((embedded, context), dim=2)

#         output, hidden = self.gru(input_gru, hidden)
#         output = self.out(output)

#         return output, hidden, attn_weights

# def indexesFromSentence(lang, sentence):
#     return [lang.word2index[word] for word in sentence.split(' ')]

# def tensorFromSentence(lang, sentence):
#     indexes = indexesFromSentence(lang, sentence)
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

# def tensorsFromPair(pair):
#     input_tensor = tensorFromSentence(input_lang, pair[0])
#     target_tensor = tensorFromSentence(output_lang, pair[1])
#     return (input_tensor, target_tensor)

# def get_dataloader(batch_size):
#     input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

#     n = len(pairs)
#     input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
#     target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

#     for idx, (inp, tgt) in enumerate(pairs):
#         inp_ids = indexesFromSentence(input_lang, inp)
#         tgt_ids = indexesFromSentence(output_lang, tgt)
#         inp_ids.append(EOS_token)
#         tgt_ids.append(EOS_token)
#         input_ids[idx, :len(inp_ids)] = inp_ids
#         target_ids[idx, :len(tgt_ids)] = tgt_ids

#     train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
#                                torch.LongTensor(target_ids).to(device))

#     train_sampler = RandomSampler(train_data)
#     train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
#     return input_lang, output_lang, train_dataloader

# def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
#           decoder_optimizer, criterion):

#     total_loss = 0
#     for data in dataloader:
#         input_tensor, target_tensor = data

#         encoder_optimizer.zero_grad()
#         decoder_optimizer.zero_grad()

#         encoder_outputs, encoder_hidden = encoder(input_tensor)
#         decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

#         loss = criterion(
#             decoder_outputs.view(-1, decoder_outputs.size(-1)),
#             target_tensor.view(-1)
#         )
#         loss.backward()

#         encoder_optimizer.step()
#         decoder_optimizer.step()

#         total_loss += loss.item()

#     return total_loss / len(dataloader)

# import time
# import math

# def asMinutes(s):
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)

# def timeSince(since, percent):
#     now = time.time()
#     s = now - since
#     es = s / (percent)
#     rs = es - s
#     return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# import matplotlib.ticker as ticker
# import numpy as np

# def showPlot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
#     # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)

# def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
#                print_every=100, plot_every=100):
#     start = time.time()
#     plot_losses = []
#     print_loss_total = 0  # Reset every print_every
#     plot_loss_total = 0  # Reset every plot_every

#     encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
#     criterion = nn.NLLLoss()

#     for epoch in range(1, n_epochs + 1):
#         loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
#         print_loss_total += loss
#         plot_loss_total += loss

#         if epoch % print_every == 0:
#             print_loss_avg = print_loss_total / print_every
#             print_loss_total = 0
#             print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
#                                         epoch, epoch / n_epochs * 100, print_loss_avg))

#         if epoch % plot_every == 0:
#             plot_loss_avg = plot_loss_total / plot_every
#             plot_losses.append(plot_loss_avg)
#             plot_loss_total = 0

#     showPlot(plot_losses)

# def evaluate(encoder, decoder, sentence, input_lang, output_lang):
#     with torch.no_grad():
#         input_tensor = tensorFromSentence(input_lang, sentence)

#         encoder_outputs, encoder_hidden = encoder(input_tensor)
#         decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

#         _, topi = decoder_outputs.topk(1)
#         decoded_ids = topi.squeeze()

#         decoded_words = []
#         for idx in decoded_ids:
#             if idx.item() == EOS_token:
#                 decoded_words.append('<EOS>')
#                 break
#             decoded_words.append(output_lang.index2word[idx.item()])
#     return decoded_words, decoder_attn
# def evaluateRandomly(encoder, decoder, n=10):
#     for i in range(n):
#         pair = random.choice(pairs)
#         print('>', pair[0])
#         print('=', pair[1])
#         output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
#         output_sentence = ' '.join(output_words)
#         print('<', output_sentence)
#         print('')
hidden_size = 128
batch_size = 32

input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(DEVICE)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(DEVICE)

train(train_dataloader, encoder, decoder, 15, print_every=5, plot_every=5)
