import os, pdb

pathEmoEval = 'Session1/dialog/EmoEvaluation/'

emotionList=[]
for file in os.listdir(pathEmoEval):
    if file.endswith(".txt"):
        with open(pathEmoEval+file) as f:
            f.readline()
            for line in f:
                line = line.strip()
                temp=[]
                if line.startswith('['):
                    parts = line.split('\t')
                    temp.append(parts[1])
                    temp.append(parts[2])
                    emotionList.append(temp)
                    # print 'file ',file,' | ',parts

# print emotionList
# pdb.set_trace()

def getWdsegParentDir(filename):
    # print "processing parent dir"
    parts = filename.split('_')[:-1]
    # print parts
    return '_'.join(parts[:])

pathForcedAlign = 'Session1/sentences/ForcedAlignment/'
wordEmotionList = []

# for i in range(len(emotionList)):
#     currentFolder = []
#     filename = pathForcedAlign + getWdsegParentDir(emotionList[i][0]+'.wdseg') + '/' + emotionList[i][0]+'.wdseg'
#     currentFolder.append(getWdsegParentDir(emotionList[i][0]+'.wdseg'))
#     with open(filename) as f:
#         f.readline()
#         currentFile = []
#         currentFile.append(emotionList[i][0])
#         for line in f:
#             parts = line.strip().split()
#             temp = []
#             if parts[0].isdigit():
#                 w = parts[3].split('(')[0]
#                 if '<' in w:
#                     continue
#                 else:
#                     temp.append(w)
#                     temp.append(emotionList[i][1])
#                     currentFile.append(temp)
#         currentFolder.append(currentFile)
#     wordEmotionList.append(currentFolder)
#
# print wordEmotionList
# pdb.set_trace()

#######folder level

for j in range(len(os.listdir(pathForcedAlign))): # 28 folders
    if os.listdir(pathForcedAlign)[j].startswith('.'):
        continue
    print os.listdir(pathForcedAlign)
    print len(os.listdir(pathForcedAlign))
    currentFolder = []
    currentFolder.append(os.listdir(pathForcedAlign)[j])
    print os.listdir(pathForcedAlign)[j]
    length = len(os.listdir(pathForcedAlign)[j])/4
    print length
    for i in range(length): # No of segments in each file
        filename = pathForcedAlign + getWdsegParentDir(emotionList[i][0]+'.wdseg') + '/' + emotionList[i][0]+'.wdseg'
        currentFile = []
        f.readline()
        currentFile.append(emotionList[i][0])
        with open(filename) as f:
            for line in f:
                parts = line.strip().split()
                temp = []
                if parts[0].isdigit():
                    w = parts[3].split('(')[0]
                    if '<' in w:
                        continue
                    else:
                        temp.append(w)
                        temp.append(emotionList[i][1])
                        currentFile.append(temp)
        currentFolder.append(currentFile)
    wordEmotionList.append(currentFolder)

print wordEmotionList
pdb.set_trace()




