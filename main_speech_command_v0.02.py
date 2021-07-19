import dtw
import stats

import os
import glob
import argparse
import matplotlib.pyplot as plt

from multiprocessing import Pool, Manager

PERCENTAGE = False
VERBOSE = False

RESULTS_ROOT_DIRECTORY = "results/"

# TODO - clean up
manager = Manager()
dataLength = manager.Value('i', 0)
progression = manager.Value('i', 0)
AUCList = manager.list()
pivotList = manager.list()

def buildTrainingList(path, testingList, validationList):
    """
    Based on SpeechCommand_v0.02 directory structure.

    Create train set from the data in path depending on the test set and the 
    validation set. The created train set will contain the rest of the dataset in 
    path that is neither in the test set, nor in the validation set.

    :param path: path to dataset
    :param testingList: array representing the test set
    :param validationList: array representing the validation set
    :return: array representing the train set
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

    Shorten data to the specified percentage by directory.
    Example:
        data =  bed/00.wav
                bed/01.wav
                cat/00.wav
                cat/01.wav
        trimData(data, percentage=0.5)
        result =    bed/bed_0.wav
                    cat/cat_0.wav

    :param data: 1D array
    :param percentage: float (0 to 1)
    :return: data trimmed to percentage
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

    Create arrays of expected outcomes based on parent directory names.
    Example:
        query =     cat/00.wav
        search =    bed/00.wav
                    bed/01.wav
                    cat/00.wav
                    cat/01.wav
        expectations =  [[0, 0], [0, 0], [0, 1], [0, 1]]

    :param queryPath: query file path
    :param searchPatternPath: string pattern for search files
    :param searchPathList: array of search file paths
    :return: arrays of expected outcomes
    """
    expectations = []
    currentDirectory = ""
    queryDirectory = queryPath.split("/")[-2]
    if searchPathList == None:
        searchFileList = sorted(glob.glob(searchPatternPath))
    else:
        searchFileList = searchPathList
    for searchFile in searchFileList:
        searchDirectory = searchFile.split("/")[-2]
        if searchDirectory != currentDirectory:
            currentDirectory = searchDirectory
        if currentDirectory == queryDirectory:
            expectations.append([[0, 1]])
        else:
            expectations.append([[0, 0]])
    return expectations

def job(query, nbThresholds=1000, findOnePerSweep=True):
    """
    TODO
    """
    queryPath = path + query

    _, sweepList, _ = dtw.runSearch(queryPath, searchPathList=trainingPathList)

    expectations = buildExpectations(queryPath, searchPathList=trainingPathList)

    AUC, pivot = stats.computeROCCurve(sweepList, expectations, nbThresholds=nbThresholds, findOnePerSweep=findOnePerSweep)
    AUCList.append(AUC)
    pivotList.append(pivot)

    if PERCENTAGE:
        progression.value += 1
        print("%.2f" % (progression.value * 100 / dataLength.value) + "%", end='\r')

def run(queryList, path, trainingPathList, nbThresholds=1000, findOnePerSweep=True):
    """
    Call job() function in parallel to search each query from the given query list 
    among all search files in the given search list.
    Return the mean AUC and the mean pivot points of all parallel searches.

    :param queryList: array of query files
    :param searchList: path to query files
    :param nbThresholds: integer (number of thresholds)
    :param findOnePerSweep: boolean, True if one found query validates a sweep
    :return: AUC, ROC curve pivots (x, y, thresholds)
    """
    dataLength.value = len(queryList)
    progression.value = 0

    pool = Pool()
    # TODO - use starmap to send more arguments
    pool.map(job, queryList)
    pool.close()
    pool.join()

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
    Save AUC and ROC pivot points to the results directory. AUC is stored in a txt 
    file and ROC pivot points are saved in a csv file.

    :param AUC: float (Area Under the Curve)
    :param pivot: pandas data frame pivot representation
    :param path: results path
    :param name: filename prefix
    :return: None
    """
    if not os.path.isdir(path):
        os.mkdir(path)
    with open(path + name + "_auc.txt", 'w+') as f:
        f.write("%s\n" % AUC)
    pivot.to_csv(path + name + "_pivots.csv", index=False)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Dynamic Time Warping')
    parser.add_argument('-r', '--resultsname', type=str, default="dtw_dylnet", help='Name of the directory containing the results')
    parser.add_argument('-t', '--trimdata', type=float, default=1.0, help='Enable trimming of test, validation and training lists to the given percentage')
    parser.add_argument('path')

    printGroup = parser.add_mutually_exclusive_group()
    printGroup.add_argument('-p', '--percentage', action='store_true', help='Enable percentage display')
    printGroup.add_argument('-v', '--verbose', action='store_true', help='Enable verbose display')
    
    args = parser.parse_args()

    PERCENTAGE = args.percentage
    trimDataPercentage = args.trimdata
    VERBOSE = args.verbose
    path = args.path
    resultsDirectoryName = args.resultsname

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

    resultsPath = RESULTS_ROOT_DIRECTORY + resultsDirectoryName.rstrip('/') + "/"

    print("Running test set...")
    AUC, pivot = run(testingList, path, trainingPathList)
    save(AUC, pivot, resultsPath, "test")

    print("Running validation set...")
    AUC, pivot = run(validationList, path, trainingPathList)
    save(AUC, pivot, resultsPath, "validation")

    print("Done")

    plt.show()
