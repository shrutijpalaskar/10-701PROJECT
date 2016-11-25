import os, pdb

# Function to handle xxx emotion, and to decide how to feed it to the RNN
def handleXXX():
    return 'neu'

# reads the emotion eval files. Extracts emotion based on word segments
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
                    #calling handleXXX for xxx emotion
                    if(parts[2] == 'xxx'):
                        newEmotion = handleXXX()
                        temp.append(newEmotion)
                    else:
                        temp.append(parts[2])
                    emotionList.append(temp)


# function to get parent directory name from file name
def getWdsegParentDir(filename):
    parts = filename.split('_')[:-1]
    return '_'.join(parts[:])

# to get corresponding word and emotion, repeat emotion for all words
pathForcedAlign = 'Session1/sentences/ForcedAlignment/'
wordEmotionList = []

for i in range(len(emotionList)):
    currentFolder = []
    filename = pathForcedAlign + getWdsegParentDir(emotionList[i][0]+'.wdseg') + '/' + emotionList[i][0]+'.wdseg'
    currentFolder.append(getWdsegParentDir(emotionList[i][0]+'.wdseg'))
    with open(filename) as f:
        f.readline()
        currentFile = []
        currentFile.append(emotionList[i][0])
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

# wordEmotionList is the final list that you should access to get words and emotions


# print wordEmotionList
# pdb.set_trace()

# allFiles = []
# for j in range(len(os.listdir(pathForcedAlign))): # 28 folders
#     if os.listdir(pathForcedAlign)[j].startswith('.'):
#         continue
#     # else:
#     #     allFiles.append(os.listdir(pathForcedAlign)[j])
#     # print allFiles
#     # print len(os.listdir(pathForcedAlign))
#     currentFolder = []
#     # currentFolder.append(allFiles)
#     # pdb.set_trace()
#     currentFolder.append(os.listdir(pathForcedAlign)[j])
#     # print currentFolder
#     length = 7
#     # print length
#     for i in range(length): # No of segments in each file
#         pdb.set_trace()
#         filename = pathForcedAlign + getWdsegParentDir(emotionList[i][0]+'.wdseg') + '/' + emotionList[i][0]+'.wdseg'
#         currentFile = []
#         # f.readline()
#         currentFile.append(emotionList[i][0])
#         with open(filename) as f:
#             for line in f:
#                 parts = line.strip().split()
#                 temp = []
#                 if parts[0].isdigit():
#                     w = parts[3].split('(')[0]
#                     if '<' in w:
#                         continue
#                     else:
#                         temp.append(w)
#                         temp.append(emotionList[i][1])
#                         currentFile.append(temp)
#         currentFolder.append(currentFile)
#     wordEmotionList.append(currentFolder)

# print wordEmotionList
# pdb.set_trace()