import spacy
import numpy

import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable
import torch.optim as optim


import argparse
import cPickle as pkl

from collections import OrderedDict

class BBNet(nn.Module):

    def __init__(self, options):
        super(BBNet,self).__init__()

        self.options = options

        self.emb = nn.Embedding(options['n_words']+1, options['n_hid'])

        self.hids = []
        for li in xrange(options['n_layers']):
            self.hids.append([
                nn.Linear(options['n_hid'], options['n_hid']),
                eval('nn.{}'.format(options['act']))()
                ])
            indim = options['n_hid']
        self.hid_modules = nn.Sequential([h[0] for h in self.hids])

        self.classifier = nn.Linear(options['n_hid'], 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, s1, s1m):
        s1emb = self.emb(s1)
        s1emb = torch.mul(s1emb, s1m.unsqueeze(2).expand_as(s1emb))
        s1emb = torch.sum(s1emb,1).squeeze()

        h = s1emb

        for li in xrange(self.options['n_layers']):
            h = self.hids[li][0](h)
            h = self.hids[li][1](h)

        z = self.classifier(h)

        return z

label_map = OrderedDict({
    '-1': 0,
    '+1': 1,
    })

class data_iterator:

    def __init__(self, fname, options):
        self.fname = fname
        self.options = options

        self.source = open(fname, 'r')
        self.source.readline() # dump the header

        self.vocab = pkl.load(open(options['vocab'], 'rb'))

        self.end_of_data = False

        self.nlp = spacy.load('en')


    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.source.readline() # dump the header

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        s1 = []
        labels = []

        try:
            while True:
                line = self.source.readline()
                if line == '':
                    raise IOError

                cols = [c.strip() for c in line.decode('utf8', errors='ignore').split('\t')]
                l_ = cols[0].lower()
                if l_ == '-':
                    continue

                if len(cols) < 3:
                    labels.append(0)
                else:
                    if cols[2] not in label_map:
                        continue
                    labels.append(label_map[cols[2]])
                s1_ = self.process(cols[1].lower())
                s1.append(s1_)

                if len(s1) > self.options['batch']:
                    break
        except IOError:
            self.end_of_data = True

        if len(s1) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        s1, s1m = self.equalizer(s1)

        return s1, s1m, labels

    def equalizer(self, sents):
        max_len = numpy.max([len(s) for s in sents])
        sents_ = []
        masks_ = []
        for sent in sents:
            s_ = [0] * max_len
            m_ = [1] * len(sent) + [0] * (max_len - len(sent))
            masks_.append(m_)
            s_[:len(sent)] = sent
            sents_.append(s_)
        return sents_, masks_

    def process(self, sent):
        sent = sent.replace('"', '')
        uwords = [t.text for t in self.nlp(unicode(sent))]
        bong = []
        for oo in xrange(1,self.options['order']+1):
            for ng in set([' '.join(t).strip() for t in 
                           zip(*[uwords[i:] for i in xrange(oo)])]):
                if ng in self.vocab:
                    idx = self.vocab[ng]
                    if idx > self.options['n_words']:
                        pass
                    else:
                        bong.append(idx)
                else:
                    pass
        return bong


def main(options):

    torch.manual_seed(options['seed'])

    ui = 0
    best_val = 0.

    train = data_iterator('./data/training_data/train.shuf.tsv', options)
    val = data_iterator('./data/training_data/val.shuf.tsv', options)

    net = BBNet(options)
    optimizer = optim.Adam(net.parameters())

    for ei in xrange(options['n_epochs']):
        print 'Starting the {}-th epoch..'.format(ei+1)

        for s1, s1m, labels in train:
            net.train()

            ui += 1

            s1_ = torch.from_numpy(numpy.array(s1))
            s1m_ = torch.from_numpy(numpy.array(s1m).astype('float32'))

            out = net(Variable(s1_), Variable(s1m_))

            target = Variable(torch.from_numpy(numpy.array(labels)))
            loss = net.criterion(out, target)

            net.zero_grad()
            loss.backward()

            optimizer.step()

            if numpy.mod(ui, options['disp_freq']) == 0:
                print 'Update {} loss {}'.format(ui+1, loss.data.numpy())

            if numpy.mod(ui, options['val_freq']) == 0:
                net.eval()
                print 'Validating...',
                n_corrects = 0
                n_samples = 0
                for s1, s1m, labels in val:
                    s1_ = torch.from_numpy(numpy.array(s1))
                    s1m_ = torch.from_numpy(numpy.array(s1m).astype('float32'))

                    out = net(Variable(s1_), Variable(s1m_))
                    out = out.data.numpy()
                    n_corrects += numpy.sum(out.argmax(-1) == labels) 
                    n_samples += len(labels)
                acc = float(n_corrects)/n_samples
                print '{}/{} (Accuracy: {})'.format(n_corrects, n_samples, acc)

                if acc > best_val:
                    best_val = acc

                    net.cpu()
                    with open('{}.best.pkl'.format(options['saveto']), 'wb') as f:
                        pkl.dump(net, f, protocol=pkl.HIGHEST_PROTOCOL)

    net.cpu()
    with open(options['saveto'], 'wb') as f:
        pkl.dump(net, f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=int, default=3)
    parser.add_argument('-n', type=int, default=100000)
    parser.add_argument('-emb', type=int, default=64)
    parser.add_argument('-hid', type=int, default=100)
    parser.add_argument('-layer', type=int, default=2)
    parser.add_argument('-act', type=str, default='Tanh')
    parser.add_argument('-batch', type=int, default=64)
    parser.add_argument('-disp-freq', type=int, default=10)
    parser.add_argument('-save-freq', type=int, default=1000)
    parser.add_argument('-val-freq', type=int, default=1000)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('vocab', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    options = OrderedDict(
            {'n_words': args.n,
             'order': args.o,
             'saveto': args.saveto,
             'vocab': args.vocab,
             'n_layers': args.layer,
             'n_hid': args.hid,
             'emb': args.emb,
             'act': args.act,
             'batch': args.batch,
             'disp_freq': args.disp_freq,
             'save_freq': args.save_freq,
             'val_freq': args.val_freq,
             'n_epochs': args.epochs,
             'seed': args.seed,
            })

    main(options)

