import lasagne
import numpy as np
import theano
import theano.tensor as T
from PIL import Image
from PROJECT.prod import WordProcessing as WP
from PROJECT.prod import DataProcessing as DP

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(11))
# Sequence Length
SEQ_LENGTH = 20
# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512
# Optimization learning rate
LEARNING_RATE = .01
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
PRINT_FREQ = 1000
# Number of epochs to train the net
NUM_EPOCHS = 50
# Batch Size
BATCH_SIZE = 128
# Number of features
NUM_FEATURES = 1752


class SpeechLSTM(object):

    def buildLSTM(self, numEpochs=NUM_EPOCHS, numFeatures=NUM_FEATURES, isBatchNormalized=False,
                  l2Regularization=0.4, mode="train"):
        """ Build model """
        inputLayer = lasagne.layers.InputLayer(shape=(None, 20, 30, 34))
        lstm1Layer = lasagne.layers.LSTMLayer(incoming=inputLayer,
                                              num_units=N_HIDDEN,
                                              grad_clipping=GRAD_CLIP,
                                              nonlinearity=lasagne.nonlinearities.tanh
                                              )
        if isBatchNormalized:
            batchNormLayer = lasagne.layers.BatchNormLayer(incoming=lstm1Layer)

        lstm2Layer = lasagne.layers.LSTMLayer(incoming=lstm1Layer,
                                              num_units=N_HIDDEN,
                                              grad_clipping=GRAD_CLIP,
                                              nonlinearity=lasagne.nonlinearities.tanh,
                                              # only_return_final=True
                                              )
        outputLayer = lasagne.layers.DenseLayer(incoming=lstm2Layer,
                                                num_units=numFeatures,
                                                W = lasagne.init.Normal(),
                                                nonlinearity=lasagne.nonlinearities.softmax
                                                )

        # SYMBOLIC Theano tensors for the targets
        # self.inputVar = T.tensor3('input')
        self.targetValues = T.ivector('target_output')
        self.prediction = lasagne.layers.get_output(outputLayer)

        # deal with loss before optimization updates
        entropyLoss = T.nnet.categorical_crossentropy(self.prediction, self.targetValues).mean()
        if (l2Regularization > 0):
            l2Loss = l2Regularization * lasagne.regularization.regularize_layer_params(
                outputLayer, lasagne.regularization.l2)
        else:
            l2Loss = 0.0

        self.loss = entropyLoss + l2Loss
        print("Computing updates ...")
        self.all_params = lasagne.layers.get_all_params(outputLayer, trainable=True)
        updates = lasagne.updates.adagrad(self.loss, self.all_params, LEARNING_RATE) # ADA_GRAD



        print("Compiling functions ...")
        if mode == "train":
            self.train = theano.function(inputs = [
                                                   #self.inputVar,
                                                   inputLayer.input_var,
                                                   self.targetValues],
                                          outputs=[self.prediction, self.loss],
                                          updates=updates,
                                          allow_input_downcast=True
                                          )
        # test
        self.test = theano.function(inputs=[inputLayer.input_var,
                                        self.targetValues],
                                       outputs=[self.prediction, self.loss],
                                       allow_input_downcast=True
                                       )
        # probs = theano.function([inputLayer.input_var],
        #                         self.prediction,
        #                         allow_input_downcast=True)


    def calculateAccuracy(self, predictions, truth):
        """
        Return accuracy of prediction against ground truth
        :param predictions: numpy matrix
        :param truth: numpy matrix
        :return: accuracy (%)
        """
        return np.mean(predictions == truth)


    # def read_batch(self, data_raw, batch_index):
    #     """ Sample of reading batch"""
    #     start_index = batch_index * self.batch_size
    #     end_index = start_index + self.batch_size
    #
    #     data = np.zeros((self.batch_size, SEQ_LENGTH, NUM_FEATURES), dtype=np.float32)
    #     answers = []
    #
    #     for i in range(start_index, end_index):
    #         answers.append(int(data_raw[i].split(',')[1]))
    #
    #         name = data_raw[i].split(',')[0]
    #         path = self.png_folder + name + ".png"
    #         im = Image.open(path)
    #         data[i - start_index, :, :] = np.transpose(np.array(im).astype(np.float32) / 256.0)
    #
    #     answers = np.array(answers, dtype=np.int32)
    #     return data, answers

