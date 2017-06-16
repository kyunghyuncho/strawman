import spacy
import numpy

import argparse
import cPickle as pkl

from collections import OrderedDict

nlp = spacy.load('en')

def extract_ngrams(vocab, sent, order=3):
    # tokenization
    uwords = [t.text for t in nlp(unicode(sent))]
    for oo in xrange(1,order+1):
        for ng in set([' '.join(t).strip() for t in zip(*[uwords[i:] for i in xrange(oo)])]):
            if ng in vocab:
                vocab[ng] += 1
            else:
                vocab[ng] = 1

    return vocab

def main(source, saveto, order=3):

    vocab0 = OrderedDict()

    with open(source, 'r') as f:
        header = f.readline()
        cols = [c.strip() for c in header.split('\t')]

        for li, line in enumerate(f):
            cols = [c.strip() for c in line.decode('utf8').split('\t')]
            vocab0 = extract_ngrams(vocab0, cols[1].lower(), order=order)

            if numpy.mod(li, 100) == 0:
                print 'Processed {} lines so far..'.format(li+1)

    tokens = vocab0.keys()
    freqs = vocab0.values()

    sidx = numpy.argsort(freqs)[::-1]
    vocab = OrderedDict([(tokens[s],i+1) for i, s in enumerate(sidx)])

    with open(saveto, 'wb') as f:
        pkl.dump(vocab, f, protocol=pkl.HIGHEST_PROTOCOL)
        pkl.dump(vocab0, f, protocol=pkl.HIGHEST_PROTOCOL) # frequency

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=int, default=3)
    parser.add_argument('source', type=str)
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    main(args.source, args.saveto, order=args.o)








