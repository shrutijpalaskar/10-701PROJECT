import os
import cPickle as pickle

import itertools
import numpy as np

# CONSTANTS
PICKLE_DIR = "../data/pickle/"
WAV_DIR = "../data/output_wavfiles/"


def getFeaturesExtracted(dir="../data/output_wavfiles", prefixes=None):
    """
    Currently it's for Session01 only, needs to do the same for Session02 -> 05
    3 levels: section -> sentence -> word + features
    Ses01F_impro01 => {
        'Ses01F_impro01_F000' => [(w:feature), (w:feature), ...]
        'Ses01F_impro01_M000' => [(w:feature), (w:feature), ...]
        ...
    }
    """
    maxLen = 0
    masterList = [] # <-> list of sentence, one next to other
    vocab = set()
    for prefix in prefixes:
        print("Processing {0}".format(prefix))
        list= [] # <-> sentence
        data = pickle.load(open(PICKLE_DIR + "dataset_" + prefix + ".txt.p", "rb"))

        for sentence in data:
            sublist = [] # word + feature
            # print sentence

            # from sentence[0] we can retrieve CSV (extracted feature)
            extractedFeatureprefix = sentence[0]
            files = [f for f in os.listdir(WAV_DIR) if f.startswith(extractedFeatureprefix) and "_st" in f]

            # now take features
            wordFeatures = [getFeaturesExtractedFromCSV(WAV_DIR + f) for f in files]

             # get the list of raw words (this will map 1:1 with the features above)
            wordsAndAlignments = [w[0] for w in sentence[3]]
            vocab.add(word for word in wordsAndAlignments)

            # update the tuple [word, extracted matrix] for every word in sentence
            sublist.append([extractedFeatureprefix, zip(wordsAndAlignments, wordFeatures)])

            # update max len
            if len(wordsAndAlignments) > maxLen:
                maxLen = len(wordsAndAlignments)

            list.append([extractedFeatureprefix, sublist])

        masterList.append([prefix, list])
    print("Longest sentence has the length of {0}".format(maxLen))

    return vocab, masterList


def getFeaturesExtractedFromCSV(filename="../data/output_wavfiles/Ses01F_impro01_F000_0.wav_st.csv"):
    features = []
    with open(filename) as f:
        for line in f:
            features.append([float(field) for field in line.strip().split(',')])
    return np.asarray(features)

def getTranscriptions(dir="../data/transcriptions"):
    """ Given transcription directory, get all the filename prefixes """
    prefix = []
    for filename in os.listdir(dir):
        prefix.append(getPrefixFromTranscription(filename))
    return prefix

def getPrefixFromTranscription(filename="Ses01F_impro01.txt"):
    """ Given a filename, get the prefix. E.g. get 'Ses01F_impro01' """
    return filename.split(".")[0]

def main():
    prefixes = getTranscriptions()
    print prefixes
    vocab, masterList = getFeaturesExtracted(prefixes=prefixes)

    fp = open('masterList.p', 'wb')
    toPickle = []
    for item in masterList:
        # print item
        toPickle.append(item)
    pickle.dump(toPickle, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # fp2 = open('vocab.p', 'wb')
    # pickle.dump(list(vocab), fp, protocol=pickle.HIGHEST_PROTOCOL)
    # print 'Pickled and Done.'

def getVocabAndMasterListFromPicke(filename="../data/masterList.p"):
    """ Given a picke file of master list, yield the master list itself, along with the vocabulary (unique words)"""
    masterList = pickle.load(open(filename, "rb"))
    masterSentences = [] # by now ignore the relation between sentences
    vocab = set()
    print len(masterList)
    maxMatrixRow = 0
    for prefixAndList in masterList:
        print("Prefix is {0}".format(prefixAndList[0]))
        for featurePrefixAndSublit in prefixAndList[1]:
            print("====Feature Prefix: " + featurePrefixAndSublit[0])
            for sentenceAndFeatures in featurePrefixAndSublit[1]:
                # print len(sentenceAndFeatures)
                print ("***")
                # print len(sentenceAndFeatures[0])
                print len(sentenceAndFeatures[1])
                print ("***")
                masterSentences.append(sentenceAndFeatures[1]) # ignore sentences relation
                for wordAndFeatures in sentenceAndFeatures[1]:
                    vocab.add(wordAndFeatures[0])
                    print wordAndFeatures[0], wordAndFeatures[1].shape, type(wordAndFeatures[1])
                    # update max row possible
                    if maxMatrixRow < wordAndFeatures[1].shape[0]:
                        maxMatrixRow = wordAndFeatures[1].shape[0]
    print("Feature matrices has the largest row of {0} (for pre-padding purpose)".format(maxMatrixRow))
    return vocab, masterList, masterSentences

def preProcessDataForRNNFromPickle(masterSentences=None, vocab=None, maxSentenceLength=20,
                                   targetNumRow=30, targetNumCol=34, trainPortion=0.8):
    """ after getting master sentence list (each sentence as tuple(words, featureMatrix)
    and vocab, we pre-process the data
    :param masterSentences a list of all sentences
    """
    np.random.seed(11)
    X, y = [], []

    vocab.add("S_START")
    vocab.add("S_STOP")
    word2idx = dict((w, id) for id, w in enumerate(vocab))
    idx2word = dict((id, w) for id, w in enumerate(vocab))

    # pad sentences with start and stop
    paddedMasterSentences = paddedAllSentences(masterSentences, targetNumCol, targetNumRow)
    allText = list(itertools.chain(*paddedMasterSentences))

    step = 10
    sentences = []
    nextWord = []
    # grab 2 words at a time, record the next one
    for i in range(0, len(allText) - maxSentenceLength, step):
        sentences.append(allText[i: i + maxSentenceLength])
        nextWord.append(allText[i + maxSentenceLength])
    print('nb sequences after extracting:', len(sentences))
    assert len(sentences) == len(nextWord)

    # print nextWord[0][0], type(nextWord[0][1]), nextWord[0][1].shape

    X, y = [], []
    for sentence in sentences:
        X.append([feature for (word, feature) in sentence])
    # y.append(word2idx[word] for (word, feature) in nextWord)
    for (word, feature) in nextWord:
        y.append(word2idx[word])

    X, y = np.asarray(X), np.asarray(y).T
    startTestIndex = (int)(len(sentences) * trainPortion)

    X_test = X[startTestIndex:]
    y_test = y[startTestIndex:]
    X_train = X[:startTestIndex]
    y_train = y[:startTestIndex]

    return X_train, y_train, X_test, y_test


def paddedAllSentences(masterSentences, targetNumCol, targetNumRow):
    print("Now pre-padding all words ")
    paddedMasterSentences = []
    for sentence in masterSentences:
        tmp = [("S_START", np.random.rand(targetNumRow, targetNumCol))]

        # normailize very word
        for (w, feature) in sentence:
            newFeature = prePaddingFeatureMatrix(feature, targetNumRow, targetNumCol)
            tmp.append((w, newFeature))
            print w, feature.shape, newFeature.shape

        tmp += [("S_STOP", np.random.rand(targetNumRow, targetNumCol))]
        paddedMasterSentences.append(tmp)
    print("Finally we have {0} sentences vs. {1} previously".format(len(paddedMasterSentences), len(masterSentences)))
    # print paddedMasterSentences[0][0][0], paddedMasterSentences[0][0][1].shape

    return paddedMasterSentences

def prePaddingFeatureMatrix(matrix=None, targetNumRow=0, targetNumCol=0):
    """ Given a matrix, padding more rows so that it will have equal size
    E.g. in our case, we have (3, 34), (1, 34), (9, 34) -> would make all to (10, 34)
    :param matrix: a numpy matrix
    :param targetNumRow is the targeted number of rows to get, this should be bigger or equal to matrix.shape[0]
    :param targetNumCol is the targeted number of cols (99% of existed matrix has this number,
        only 1% has (0,) so they need this
    :return pre-padded matrix

    UPDATE: words for row trimming as well, e.g. (213, 34) -> (30, 34)
    """

    if matrix.shape[0] == 0: # for matrix of shape (0, )
        return np.zeros((targetNumRow, targetNumCol), dtype=float)

    if matrix.shape[0] < targetNumRow:
        numRowsToAdd = targetNumRow - matrix.shape[0]
        matrixToAdd = np.zeros((numRowsToAdd, targetNumCol), dtype=float)
        return np.concatenate((matrixToAdd, matrix), axis=0)
    else:
        step = matrix.shape[0] / targetNumRow
        matrixToAdd = matrix[0, :].reshape(1, targetNumCol)
        for i in range(step, matrix.shape[0], step):
            matrixToAdd = np.concatenate((matrixToAdd, matrix[i, :].reshape(1, targetNumCol)), axis=0)
            if (matrixToAdd.shape[0] == targetNumRow):
                break
        return matrixToAdd.reshape(targetNumRow, targetNumCol)

# TEST
if __name__ == "__main__":
    # main() # do not do this multiple tiles, they are dumped to pickle
    vocab, masterList, masterSentences = getVocabAndMasterListFromPicke()
    print("Vocab size is {0} and there are total {1} sentences".format(len(vocab), len(masterSentences)))

    X_train, y_train, X_test, y_test = preProcessDataForRNNFromPickle(masterSentences, vocab)
    print X_train.shape, y_train.shape
    print X_test.shape, y_test.shape

