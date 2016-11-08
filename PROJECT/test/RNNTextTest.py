from PROJECT.prod import LSTMText as RNN
from PROJECT.prod import WordProcessing as WP


# TEST DRIVER
def main():
    wordProcessing = WP.WordsProcessing()
    print wordProcessing.content
    print wordProcessing.vocab
    print wordProcessing.word2idx
    print wordProcessing.idx2word

    # print wordProcessing.embedding("director")
    X_train, y_train, X_test, y_test = wordProcessing.buildTrainData()
    print X_train.shape, y_train.shape
    print X_test.shape, y_test.shape

    rnn = RNN.TextLSTM()
    rnn.buildLSTM(numFeatures=len(wordProcessing.word2idx))
    # X_train = np.reshape(X_train, X_train.shape + (1,))
    # y_train = np.reshape(y_train, y_train.shape + (1,))
    print X_train.shape, y_train.shape
    # Minibatch size
    MINIBATCH_SIZE = 100
    NUM_EPOCHS = 50
    for i in range(NUM_EPOCHS):
        # for _ in range(MINIBATCH_SIZE):

        predictions, cost = rnn.train(X_train, y_train)
        print cost



if __name__ == "__main__":
    main()