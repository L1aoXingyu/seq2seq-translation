import time

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TextDataset
from model.seq2seq import AttnDecoderRNN, DecoderRNN, EncoderRNN

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20
lang_dataset = TextDataset()

lang_dataloader = DataLoader(lang_dataset, shuffle=True)

input_size = lang_dataset.input_lang_words
hidden_size = 256
output_size = lang_dataset.output_lang_words
total_epoch = 100

encoder = EncoderRNN(input_size, hidden_size)
decoder = DecoderRNN(hidden_size, output_size, n_layers=2)

if torch.cuda.is_available():
    encoder = encoder.cuda()
    decoder = decoder.cuda()
param = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(param, lr=1e-2)
criterion = nn.NLLLoss()

for epoch in range(total_epoch):
    since = time.time()
    loss = 0
    for i, data in enumerate(lang_dataloader):
        in_lang, out_lang = data
        in_lang = Variable(in_lang)
        out_lang = Variable(out_lang)

        encoder_hidden = encoder.initHidden()
        encoder_output, encoder_hidden = encoder(in_lang, encoder_hidden)

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()
        decoder_hidden = encoder_hidden
        for di in range(out_lang.size(1)):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            loss += criterion(decoder_output[0], out_lang[di])
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
            if ni == EOS_token:
                break
        if (i + 1) % 1000 == 0:
            print('Loss:{:.6f}'.format(loss.data[0] / (i + 1)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    during = time.time() - since
    print('Finish {}/{} , Loss:{:.6f}, Time:{:.0f}'.format(
        epoch + 1, total_epoch), loss.data[0] / len(lang_dataset), during)
