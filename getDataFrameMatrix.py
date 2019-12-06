from c3dReadClass import C3DData 
from walkingSubjects import subjectToTrialMap

ALL_C3D_DIR = "./data/allc3d/subjects/"

# this is how the file paths are
def getPaddedNumberToString(subjectOrTrialNumber):
    dirName = str(subjectOrTrialNumber)
    if len(dirName) < 2:
        dirName = '0' + dirName
    return dirName

def readAllC3dFiles(subjectToTrialMap):
    subjectsData = {}
    for subjectNumber in subjectToTrialMap.keys():
        dirPath = ALL_C3D_DIR + getPaddedNumberToString(subjectNumber) + "/"
        for trialNumber in subjectToTrialMap[subjectNumber]:
            filePath = dirPath + getPaddedNumberToString(subjectNumber) + "_" + getPaddedNumberToString(trialNumber) + ".c3d"
            # print(filePath)
            try:
                subjectsData[subjectNumber] = C3DData(None, filePath)
            except:
                print("file not found")
    # print("files read length: " + str(len(subjectsData)))
    return subjectsData


if __name__ == "__main__":
    print("Subject and trial numbers used: ")
    print(subjectToTrialMap)
    print("\n")
    allSubjectsData = readAllC3dFiles(subjectToTrialMap)
    markersPerSubject = {}
    allSubjectsMarkerFrames = {}
    
    # if(allSubjectsData[0].Data['AllPoints'].keys() == allSubjectsData[0].Data['Markers'].keys()):
    #     print("same keys")
    
    for subject in allSubjectsData.keys():
        thisSubjectMarkerData = allSubjectsData[subject].Data['Markers']
        # thisSubjectAllData = allSubjectsData[subject].Data['Markers'];
        thisSubjectMetadata = allSubjectsData[subject].Gen
        firstFrame = thisSubjectMetadata['Vid_FirstFrame']
        lastFrame = thisSubjectMetadata['Vid_LastFrame']
        markersPerSubject[subject] = thisSubjectMarkerData.keys()
        # print("no of markers = " + str(len(markersPerSubject[subject])) )
        thisSubjectMarkerFrames = {}
        for markerKey in markersPerSubject[subject]:
            # print(markerKey)
            strippedMarkerKey = markerKey.split(':')[1]
            thisSubjectMarkerFrames[strippedMarkerKey] = thisSubjectMarkerData[markerKey]
        allSubjectsMarkerFrames[subject] = thisSubjectMarkerFrames

    print("For marker = 'TakeoMonday:RFWT' ")
    print(allSubjectsMarkerFrames[2]['RFWT'][2])
    print(allSubjectsMarkerFrames.keys())
    for subject in allSubjectsMarkerFrames.keys():
        print(allSubjectsMarkerFrames[subject].keys())

