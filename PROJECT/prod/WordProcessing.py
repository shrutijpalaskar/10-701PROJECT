import itertools
import numpy as np

START_TOKEN = 'S_START'
END_TOKEN = 'S_STOP'

class WordsProcessing(object):

    def __init__(self, filename="../data/text.txt"):
        """
        Constructor
        :param filename: text file
        """
        # self.content = []
        with open(filename) as f:
                self.content = [self.processLine(line) for line in f if line != '\n']

        self.vocab = set(itertools.chain(*self.content))
        print 'There are {0} unique words in vocabulary'.format(len(self.vocab))

        self.word2idx = dict((w, id) for id, w in enumerate(self.vocab))
        self.idx2word = dict((id, w) for id, w in enumerate(self.vocab))

    def processLine(self, line):
        line = START_TOKEN + " " + line.decode('utf-8').lower().strip() + " " + END_TOKEN
        line = line.replace('\'s', '')
        return line.split()

    def buildTrainData(self, trainPortion=0.8, maxSentenceLength=35):
        step = 3
        sentences = []
        nextWord = []
        # grab 3 words at a time, record the next one
        allText = list(itertools.chain(*self.content))
        for i in range(0, len(allText) - maxSentenceLength, step):
            sentences.append(allText[i: i + maxSentenceLength])
            nextWord.append(allText[i + maxSentenceLength])
        print('nb sequences after extracting:', len(sentences))
        assert len(sentences) == len(nextWord)

        print ('last sentence len is: {0}').format(len(sentences[-1]))
        print ('last word is: {0}').format(nextWord[-1])

        # build training data
        oneHotEmbeddingDimension = len(self.word2idx)
        X_data = np.zeros((len(sentences), maxSentenceLength, oneHotEmbeddingDimension), dtype=int)
        y_data = np.zeros((len(nextWord), oneHotEmbeddingDimension), dtype=int)

        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence):
                X_data[i, t] = self.embedding(word)
            y_data[i] = self.embedding(nextWord[i])

        # print ('X and y content: ')
        # print (X_train.shape, y_train.shape)
        # print (X_train[0][3])
        # print (y_train[0])

        startTestIndex = (int)(len(sentences) * trainPortion)
        X_test = X_data[startTestIndex:]
        y_test = y_data[startTestIndex:]
        X_train = X_data[:startTestIndex]
        y_train = y_data[:startTestIndex]

        return X_train, y_train, X_test, y_test

    def embedding(self, word):
        """ Embedding a word into a hot vector"""
        dimension = len(self.vocab) # one hot vector of this length
        embedded = np.zeros(dimension, dtype=int)
        embedded[self.word2idx[word]] = 1
        return embedded