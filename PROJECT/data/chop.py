__author__ = 'me'

import cPickle as pickle
from subprocess import call


fp = open('dataset.p', "rb")
data = pickle.load(fp)

print len(data)
for i in range(len(data)):
    print data[i]

    # print 'len data[i][3]', len(data[i][3])
    for j in range(len(data[i][3])):
        newName = data[i][0]+'_'+str(j)+'.wav'
        # print newName
        # print data[i][3][j][0], data[i][3][j][1]
        # print data[i][3][j][0], data[i][3][j][2]
        call(["sox", "Ses01M_script03_2.wav", newName, "trim", str(data[i][3][j][1]), str(data[i][3][j][2] - data[i][3][j][1])])
        if i==4:
            break