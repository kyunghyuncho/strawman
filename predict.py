import argparse
import cPickle as pkl

import numpy

import torch
from torch import nn
from torch.autograd import Variable

from trainer import data_iterator, BBNet

SNLINet = BBNet

def main(loadfrom, saveto, dev_data):

    loadfrom = loadfrom.strip().split(',')

    net = []
    for ll in loadfrom:
        with open(ll, 'rb') as f:
            net.append(pkl.load(f))

    test = data_iterator(dev_data, net[0].options)

    print 'Testing...',
    preds = []
    n_samples = 0
    softmax = torch.nn.Softmax()
    for s1, s1m, labels in test:
        for nn in net:
            nn.train()
        s1_ = torch.from_numpy(numpy.array(s1))
        s1m_ = torch.from_numpy(numpy.array(s1m).astype('float32'))

        for ii, nn in enumerate(net):
            out = nn(Variable(s1_,requires_grad=False), 
                     Variable(s1m_,requires_grad=False))
            out = softmax(out)
            out = out.data.numpy()
            if ii == 0:
                pp = out
            else:
                pp += out
        pp = pp / len(net)

        preds.append(pp.argmax(-1))
        n_samples += len(labels)

    preds = numpy.concatenate(preds, axis=0)
    preds = (2. * preds) - 1.

    pos = numpy.sum(preds == 1.)
    neg = numpy.sum(preds == -1.)
    print 'pos {} neg {}'.format(pos, neg)


    with open(saveto, 'w') as f_out:
        with open(dev_data, 'r') as f_in:
            print >>f_out, f_in.readline().strip()
            for ii, l in enumerate(f_in):
                print >>f_out, '{}\t{}'.format(l.strip(), int(preds[ii]))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_data', type=str, default='./data/dev_data/blind_dev_sentences.tsv')
    parser.add_argument('loadfrom', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.loadfrom, args.saveto, args.dev_data)

