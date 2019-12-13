from c3dReadClass import C3DData 
from walkingSubjects import subjectToTrialMap as walkingSubjectsMap
from jumpingSubjects import subjectToTrialMap as jumpingSubjectsMap
from jumpingSubjects import runningSubjectToTrialMap as runningSubjectsMap
from jumpingSubjects import otherSubjectToTrialMap

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
import numpy as np
from numpy import mean
from numpy import std

ALL_C3D_DIR = "./data/allc3d/subjects/"

# this is how the file paths are
def getPaddedNumberToString(subjectOrTrialNumber):
    dirName = str(subjectOrTrialNumber)
    if len(dirName) < 2:
        dirName = '0' + dirName
    return dirName

def readAllC3dFiles(subjectToTrialMap):
    
    ## to record errors
    fileNotFound = []
    
    subjectsDataArray = []
    for subjectNumber in subjectToTrialMap.keys():
        dirPath = ALL_C3D_DIR + getPaddedNumberToString(subjectNumber) + "/"
        for trialNumber in subjectToTrialMap[subjectNumber]:
            filePath = dirPath + getPaddedNumberToString(subjectNumber) + "_" + getPaddedNumberToString(trialNumber) + ".c3d"
            # print(filePath)
            try:
                c3dBinaryFile = C3DData(None, filePath)
                subjectsDataArray.append(c3dBinaryFile)
            except:
                fileNotFound.append("file not found, subjectNum: " + str(subjectNumber) + " ,trial: " + str(trialNumber) + " filePath: " + filePath )
    
    with open("errorFile.log", 'a') as f:
        for errorLine in fileNotFound:
            f.write(errorLine + '\n')

    return subjectsDataArray



def performTrainTestSplit(allTrialData, allOutputLabels, walkingSubjectsCount, otherSubjectsCount):
    ## do train test split

    print("walking sub: " + str(walkingSubjectsCount))
    print("other sub: " + str(otherSubjectsCount))

    trainWalkCount = int(0.8*walkingSubjectsCount)
    trainOtherCount = int(0.8*otherSubjectsCount)
    

    print("walking train: " + str(trainWalkCount))
    print("other train: " + str(trainOtherCount))

    trainX = np.concatenate( (allTrialData[0:trainWalkCount], allTrialData[walkingSubjectsCount: walkingSubjectsCount+trainOtherCount]), axis=0 )
    testX = np.concatenate( (allTrialData[trainWalkCount: (walkingSubjectsCount-trainWalkCount)], allTrialData[walkingSubjectsCount: walkingSubjectsCount+(otherSubjectsCount-trainOtherCount) ]), axis=0 )

    trainY = np.concatenate( (allOutputLabels[0:trainWalkCount], allOutputLabels[walkingSubjectsCount: walkingSubjectsCount+trainOtherCount]), axis=0)
    testY = np.concatenate( (allOutputLabels[trainWalkCount: (walkingSubjectsCount-trainWalkCount)], allOutputLabels[walkingSubjectsCount: walkingSubjectsCount+(otherSubjectsCount-trainOtherCount)]), axis=0 )


    return trainX, trainY, testX, testY

def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 0, 15, 64
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy


# run an experiment
def run_experiment(trainX, trainY, testX, testY, repeats=10):
	# load data
	# trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainY, testX, testY)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)


if __name__ == "__main__":
    # print("Subject and trial numbers used: ")
    # print(walkingSubjectsMap)
    print("\n")
    with open("errorFile.log", 'w') as f:
        f.write("New logs" + '\n')

    walkingSubjectsDataArray = readAllC3dFiles(walkingSubjectsMap)
    print("walking: " + str(len(walkingSubjectsDataArray)) )
    jumpingSubjectsDataArray = readAllC3dFiles(jumpingSubjectsMap)
    print("jumping: " + str(len(jumpingSubjectsDataArray)) )
    runningSubjectsDataArray = readAllC3dFiles(runningSubjectsMap)
    print("running: " + str(len(runningSubjectsDataArray)) )
    otherSubjectsDataArray = readAllC3dFiles(otherSubjectToTrialMap)
    print("other activities: " + str(len(otherSubjectsDataArray)) )
    allSubjectsDataArray = walkingSubjectsDataArray + jumpingSubjectsDataArray + runningSubjectsDataArray + otherSubjectsDataArray
    # allSubjectsDataArray = walkingSubjectsDataArray + jumpingSubjectsDataArray
    markersPerSubject = {}
    allTrialsMarkerFrames = []
    

    totalTrialsLoaded = len(allSubjectsDataArray)
    print("no of trials: " + str(totalTrialsLoaded) )


    markerCount = {}
    commonMarkersArray = []
    commonMarkersMap = {}
    minimumFrameCount = 9999


    for trial in allSubjectsDataArray:
        thisSubjectMarkerData = trial.Data['Markers']
        markersThisTrial = thisSubjectMarkerData.keys()
        thisSubjectMarkerFrames = {}
        for markerKey in markersThisTrial:
            # print("marker is: " + markerKey + " and type is: " + str(type(markerKey)))
            thisTrialFrameCount = thisSubjectMarkerData[markerKey].shape[1]
            if(thisTrialFrameCount < minimumFrameCount):
                minimumFrameCount = thisTrialFrameCount
            try:
                strippedMarkerKey = markerKey.split(':')[1]
                thisSubjectMarkerFrames[strippedMarkerKey] = thisSubjectMarkerData[markerKey]
                # print("shape: " + str(thisSubjectMarkerData[markerKey].shape) )

                if(markerCount.get(strippedMarkerKey)==None):
                    markerCount[strippedMarkerKey] = 1
                else:
                    markerCount[strippedMarkerKey] += 1
                    if(markerCount[strippedMarkerKey] == totalTrialsLoaded):
                        commonMarkersArray.append(strippedMarkerKey)
                        commonMarkersMap[strippedMarkerKey] = True

            except:
                # print("cannot split: " + markerKey )
                strippedMarkerKey = markerKey
        allTrialsMarkerFrames.append(thisSubjectMarkerFrames)

    
    print("common markers count: " + str(len(commonMarkersArray)) )
    print(commonMarkersArray)
    print("minimum frames count: " + str(minimumFrameCount) )

    # frame: 296
    # markers: 41
    # dimension per marker: 3
    # per Frame dimensions : 41 * 3 = 123
    # split into 128 frames windows, 
    #   then per window dimensions = 123 * 128 = 15,744

    # for thisTrialFrames in allTrialsMarkerFrames:
    #     for marker in thisTrialFrames.keys():
    #         print(thisTrialFrames[marker].shape, end="")
    #     print("\n")

        # create [samples, time_steps, features]
    allTrialData = list()

    # TODO: switch back to 128 frames, some example has only 127 frames, or just use the minimum no of frames encountered so far
    # [samples, 128, 123]
    # iterate the array of frames
    for trialMap in allTrialsMarkerFrames:
        # iterate the map of markerKey -> frames
        allMarkerFrames = np.zeros((127, 0))
        for markerKey in trialMap:
            if(commonMarkersMap.get(markerKey)!=None):
    #             print(trialMap[markerKey].shape)
                perMarkerFrames = trialMap[markerKey].transpose()[0:127, 0:3]
                # print("shape: " + str(perMarkerFrames.shape) )
                allMarkerFrames = np.concatenate((allMarkerFrames, perMarkerFrames), 1)
            # else:
                # print("marker key not found: " + markerKey)
    #     print(allMarkerFrames.shape)
        allTrialData.append(allMarkerFrames)
    allTrialData = np.asarray(allTrialData)
    print("all trial data " + str(allTrialData.shape))
    

    walkingSubjectsCount = len(walkingSubjectsDataArray)
    otherSubjectsCount = totalTrialsLoaded - walkingSubjectsCount
    allOutputLabels = [0] * walkingSubjectsCount + [1] * otherSubjectsCount
    print("output shape: " + str(len(allOutputLabels)) )
    # 0 - walking, 1 - not walking

    ## do train test split
    trainX, trainY, testX, testY = performTrainTestSplit(allTrialData, allOutputLabels, walkingSubjectsCount, otherSubjectsCount)

    # testY = keras.utils.to_categorical(testY)
    # trainY = keras.utils.to_categorical(trainY)
    testYTrans = keras.utils.to_categorical(testY)
    trainYTrans = keras.utils.to_categorical(trainY)

    run_experiment(trainX, trainYTrans, testX, testYTrans)