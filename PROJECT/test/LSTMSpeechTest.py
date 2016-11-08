from PROJECT.prod import LSTMText as RNN
from PROJECT.prod import WordProcessing as WP
from PROJECT.prod import DataProcessing as DP
from PROJECT.prod import LSTMSpeech as LSTM

# TEST DRIVER
def main():
    vocab, masterList, masterSentences = DP.getVocabAndMasterListFromPicke()
    print("Vocab size is {0} and there are total {1} sentences".format(len(vocab), len(masterSentences)))

    X_train, y_train, X_test, y_test = DP.preProcessDataForRNNFromPickle(masterSentences, vocab)
    print X_train.shape, y_train.shape
    print X_test.shape, y_test.shape

    rnn = LSTM.SpeechLSTM()
    rnn.buildLSTM()
    # print X_train.shape, y_train.shape
    # # Minibatch size
    # MINIBATCH_SIZE = 100
    NUM_EPOCHS = 50
    for i in range(NUM_EPOCHS):
    #     # for _ in range(MINIBATCH_SIZE):
    #
        predictions, cost = rnn.train(X_train, y_train)
        print cost

if __name__ == "__main__":
    main()