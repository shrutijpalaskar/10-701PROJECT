__author__ = 'me'

import pdb
import os
import cPickle as pickle
from collections import OrderedDict

wdsegDir = 'ForcedAlignment/'
transcriptionDir = 'transcriptions/'
def processWdseg(start, filename):
    path = wdsegDir+getWdsegParentDir(filename)+'/'+filename+'.wdseg'
    segment = []
    with open(path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if parts[0].isdigit():
                s = float(parts[0])/1000
                e = float(parts[1])/1000
                w = parts[3].split('(')[0]
                # print 's', s
                # print 'e', e
                first = start+s
                second = start+e
                # print first, second
                segment.append([w, first, second])
    return segment

def getWdsegParentDir(filename):
    # print "processing parent dir"
    parts = filename.split('_')[:-1]
    # print parts
    return '_'.join(parts[:])

def shrutiProcessParts(fullfilename):
    sentenceSegment = []
    with open(fullfilename) as f:
        for line in f:
            parts = line.strip().split()
            if not parts[0].startswith('Ses'):
                continue
            filename = parts[0]
            timestamp = processTimeStamp(parts[1])
            sentence = ' '.join(parts[2:])
            sentenceSegment.append([filename, timestamp, sentence])
    return sentenceSegment

def processTimeStamp(t='[006.2901-008.2357]:'):
    t = t.split(']')[0]
    t = t.split('[')[1]
    t = t.split('-')
    return [t[0], t[1]]

def processMaster(dirName=transcriptionDir):
    sentenceSegment = None
    for fullfilename in os.listdir(dirName):
        # print fullfilename
        sentenceSegment = shrutiProcessParts(dirName+fullfilename)
        # print sentenceSegment

        for item in sentenceSegment:
            filename = item[0]
            start = float(item[1][0])
            segment = processWdseg(start, filename)
            item.append(segment)
    return sentenceSegment
# segment = processWdseg(006.2901, 'Ses01F_impro01_F000')
# print segment
# sentenceSegment = shrutiProcessParts('transcriptions/Ses01F_impro01.txt')
# print sentenceSegment

fp = open('dataset.p', 'wb')
toPickle=[]
for item in processMaster():
    print item
    toPickle.append(item)
pickle.dump(toPickle, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # f.write(item)
print 'Pickled and Done.'