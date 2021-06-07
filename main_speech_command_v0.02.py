import dtw
import stats

import os
import glob
import argparse
import matplotlib.pyplot as plt

GRAPH = False
PERCENTAGE = False
STATS = False
VERBOSE = False

def buildTrainingList(path, testingList, validationList):
    """
    Based on SpeechCommand_v0.02 directory structure.
    """
    result = []
    walkTuple = os.walk(path)
    directoryList = [x[1] for x in walkTuple][0]
    directoryList = [x for x in directoryList if "_background_noise_" not in x]
    for directory in directoryList:
        fileList = os.listdir(path + directory)
        for wavFile in fileList:
            wavSmallPath = directory + "/" + wavFile
            if wavSmallPath not in testingList and wavSmallPath not in validationList:
                result.append(wavSmallPath)
    return result

def trimData(data, percentage=0.3):
    """
    Based on SpeechCommand_v0.02 directory structure.
    """
    lengthList = []
    currentDirectoryLength = 0
    currentDirectory = data[0].split("/")[-2]
    for element in data:
        elementDirectory = element.split("/")[-2]
        if elementDirectory != currentDirectory:
            lengthList.append(currentDirectoryLength)
            currentDirectory = elementDirectory
            currentDirectoryLength = 1
        else:
            currentDirectoryLength += 1
    lengthList.append(currentDirectoryLength)

    reducedLengthList = []
    for length in lengthList:
        reducedLengthList.append(int(length * percentage))
    
    result = []
    currentDirectoryIndex = 0
    currentDirectory = data[0].split("/")[-2]
    cpt = 0
    for i, element in enumerate(data):
        elementDirectory = element.split("/")[-2]
        if elementDirectory != currentDirectory:
            currentDirectory = elementDirectory
            currentDirectoryIndex += 1
            cpt = 1
        else:
            if cpt < reducedLengthList[currentDirectoryIndex]:
                result.append(data[i])
            cpt += 1
    return result

def buildExpectations(queryPath, searchPatternPath="", searchPathList=None):
    """
    Based on SpeechCommand_v0.02 directory structure.
    """
    expectations = []
    currentDirectory = ""
    queryFilename = queryPath.split("/")[-1]
    queryDirectory = queryPath.split("/")[-2]
    queryCode = queryFilename.split("_")[0]
    if searchPathList == None:
        searchFileList = sorted(glob.glob(searchPatternPath))
    else:
        searchFileList = searchPathList
    for searchFile in searchFileList:
        searchFilename = searchFile.split("/")[-1]
        searchDirectory = searchFile.split("/")[-2]
        searchCode = searchFilename.split("_")[0]
        if searchDirectory != currentDirectory:
            currentDirectory = searchDirectory
        if searchCode == queryCode:
            if currentDirectory == queryDirectory:
                expectations.append([[0, 1]])
            else:
                expectations.append([[0, 0]])
    return expectations

def run(data, path, trainingPathList, nbThresholds=1000, oneWord=True):
    """
    TODO
    """
    AUCList = []
    pivotList = []
    for i, query in enumerate(data):
        queryPath = path + query

        _, sweepList, _ = dtw.runSearch(queryPath, searchPathList=trainingPathList)

        expectations = buildExpectations(queryPath, searchPathList=trainingPathList)

        if STATS:
            AUC, pivot = stats.computeROCCurve(sweepList, expectations, nbThresholds=nbThresholds, oneWord=oneWord)
            AUCList.append(AUC)
            pivotList.append(pivot)

        if PERCENTAGE:
            print("%.2f" % (i * 100 / len(data)) + "%", end='\r')

    assert(len(AUCList) == len(pivotList))
    sumAUC = AUCList[0]
    sumPivot = pivotList[0]
    for i in range(1, len(AUCList)):
        sumAUC += AUCList[i]
        sumPivot += pivotList[i]
    meanAUC = sumAUC / len(AUCList)
    meanPivot = sumPivot / len(pivotList)

    return meanAUC, meanPivot

def save(AUC, pivot, path, name):
    """
    TODO
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    with open(path + name + "_auc.txt", 'w+') as f:
        f.write("%s\n" % AUC)
    pivot.to_csv(path + name + "_pivots.csv", index=False)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Dynamic Time Warping')
    parser.add_argument('-g', '--graph', action='store_true', help='Enable graph display')
    parser.add_argument('-s', '--stats', action='store_true', help='Enable statistics display')
    parser.add_argument('-t', '--trimdata', type=float, default=1.0, help='Enable trimming of test, validation and training lists to the given percentage')
    parser.add_argument('path')

    printGroup = parser.add_mutually_exclusive_group()
    printGroup.add_argument('-p', '--percentage', action='store_true', help='Enable percentage display')
    printGroup.add_argument('-v', '--verbose', action='store_true', help='Enable verbose display')
    
    args = parser.parse_args()

    GRAPH = args.graph
    PERCENTAGE = args.percentage
    STATS = args.stats
    trimDataPercentage = args.trimdata
    VERBOSE = args.verbose
    path = args.path

    dtw.PERCENTAGE = PERCENTAGE
    dtw.VERBOSE = VERBOSE
    stats.VERBOSE = VERBOSE

    path = path.rstrip('/') + "/"
    with open(path + "testing_list.txt") as f:
        testingList = f.read().splitlines()
    with open(path + "validation_list.txt") as f:
        validationList = f.read().splitlines()
    trainingList = buildTrainingList(path, testingList, validationList)

    if trimDataPercentage < 1.0:
        testingList = trimData(testingList, trimDataPercentage)
        validationList = trimData(validationList, trimDataPercentage)
        trainingList = trimData(trainingList, trimDataPercentage)

    trainingPathList = [path + x for x in trainingList]

    if STATS and GRAPH:
        figure = plt.figure()

    print("Running test set...")
    AUC, pivot = run(testingList, path, trainingPathList)
    save(AUC, pivot, "results/", "dtw_test")

    print("Running validation set...")
    AUC, pivot = run(validationList, path, trainingPathList)
    save(AUC, pivot, "results/", "dtw_validation")

    plt.show()
