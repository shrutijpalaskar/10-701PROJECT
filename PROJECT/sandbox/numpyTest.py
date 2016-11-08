import numpy as np
from PROJECT.prod import LSTMText
from PROJECT.prod import DataProcessing as DP

X_train = np.zeros((5, 6, 7), dtype=float)
print X_train.shape

a = X_train[:3]
b = X_train[3:]

print a.shape
print b.shape

print (int)(2.1)

# compare 2 numpy array
a = np.asarray([1, 0, 1])
b = np.asarray([1, 1, 1])
c = np.asarray([1, 0, 1])
print np.array_equal(a, b)
print np.array_equal(a, c)

d = np.asarray([1, 0, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5]).reshape(4, 3)
e = np.asarray([1, 0, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6]).reshape(4, 3)
print d, d.shape
print e, d.shape

rnn = LSTMText.TextLSTM()
print rnn.calculateAccuracy(d, e)

print type(zip([0, 1], ['a', 'b']))


print DP.prePaddingFeatureMatrix(matrix=d, targetNumCol=3, targetNumRow=10)
f = np.concatenate((d, e), axis=0)
print f, f.shape
print
print DP.prePaddingFeatureMatrix(matrix=f,targetNumRow=3, targetNumCol=3)