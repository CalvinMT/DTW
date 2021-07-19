import dtw
import stats

import os
import re
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Pool, Manager
from unidecode import unidecode

PERCENTAGE = False
VERBOSE = False

RESULTS_ROOT_DIRECTORY = "results/"

# TODO - clean up
manager = Manager()
queryListLength = manager.Value('i', 0)
progression = manager.Value('i', 0)
AUCList = manager.list()
pivotList = manager.list()

def buildExpectations(queryPath, searchPatternPath="", searchPathList=None, sweepStep=3, sequenced=False, useDirectoryName=False):
    """
    Based on DyLNet directory structure.

    Create arrays of expected outcomes based on parent directory names.

    :param queryPath: query file path
    :param searchPatternPath: string pattern for search files
    :param searchPathList: array of search file paths
    :param sweepStep: integer (found in Kamper's SpeechDTW)
    :param sequenced: boolean, True if sweeps are sequenced with begining and ends as from 
    transcript files
    :param useDirectoryName: boolean, True if query are to be grouped by their 
    parent directory name
    :return: arrays of expected outcomes
    """
    expectations = []
    if useDirectoryName:
        queryWord = unidecode(' '.join(re.findall('([A-Za-z]+)', queryPath.split("/")[-2])))
    else:
        queryWord = unidecode(' '.join(re.findall('([A-Za-z]+)', queryPath.split("/")[-1].split("_")[0])))
    if searchPathList == None:
        searchFileList = sorted(glob.glob(searchPatternPath))
    else:
        searchFileList = searchPathList
    for i, searchFile in enumerate(searchFileList):
        expectations.append([])
        tmp = searchFile.rstrip(".wav").split("_")
        transcriptFile = tmp[0] + "_" + tmp[1] + "_VB.ods"
        begin = int(tmp[2])
        end = int(tmp[3])
        if not os.path.isfile(transcriptFile):
            print("ODS file not found: " + transcriptFile)
            continue
        transcript = pd.read_excel(transcriptFile, names=["begin", "end", "text"], engine="odf")
        for row in transcript.itertuples():
            if row.begin > end:
                break
            if row.end < begin:
                continue
            if queryWord in unidecode(row.text):
                if sequenced:
                    expectations[i].append([[0, row.end - row.begin], 1])
                else:
                    for j in range(int(row.begin / sweepStep), int(row.end / sweepStep)):
                        expectations[i].append([j, 1])
            else:
                if sequenced:
                    expectations[i].append([[0, row.end - row.begin], 0])
                else:
                    for j in range(int(row.begin / sweepStep), int(row.end / sweepStep)):
                        expectations[i].append([j, 0])
    return expectations

def job(queryPath, searchList, nbThresholds=1000, findOnePerSweep=False, sequenced=False, useDirectoryName=False):
    """
    TODO

    :param queryPath: query file path
    :param searchList: array of search file paths
    :param nbThresholds: integer (number of thresholds)
    :param findOnePerSweep: boolean, True if one found query validates a sweep
    :param sequenced: boolean, True if sweeps are sequenced with begining and ends as from 
    transcript files
    :param useDirectoryName: boolean, True if query are to be grouped by their 
    parent directory name
    :return: None
    """
    searchListWithoutQuery = []
    query = queryPath.split("/")[-1]
    # Remove search file from which query comes from
    for search in searchList:
        if query.rstrip(".wav").split("_", 1)[1] not in search:
            searchListWithoutQuery.append(search)

    _, sweepList, _ = dtw.runSearch(queryPath, searchPathList=searchListWithoutQuery)

    expectations = buildExpectations(queryPath, searchPathList=searchListWithoutQuery, sequenced=sequenced, useDirectoryName=useDirectoryName)

    AUC, pivot = stats.computeROCCurve(sweepList, expectations, nbThresholds=nbThresholds, findOnePerSweep=findOnePerSweep, sequenced=sequenced)
    AUCList.append(AUC)
    pivotList.append(pivot)

    if PERCENTAGE:
        progression.value += 1
        print("%.2f" % (progression.value * 100 / queryListLength.value) + "%", end='\r')

def run(queryList, searchList, nbThresholds=1000, findOnePerSweep=False, sequenced=False, useDirectoryName=False):
    """
    Call job() function in parallel to search each query from the given query list 
    among all search files in the given search list.
    Return the mean AUC and the mean pivot points of all parallel searches.

    :param queryList: array of query file paths
    :param searchList: array of search file paths
    :param nbThresholds: integer (number of thresholds)
    :param findOnePerSweep: boolean, True if one found query validates a sweep
    :param sequenced: boolean, True if sweeps are sequenced with begining and ends as from 
    transcript files
    :param useDirectoryName: boolean, True if query are to be grouped by their 
    parent directory name
    :return: AUC, ROC curve pivots (x, y, thresholds)
    """
    if PERCENTAGE:
        print("0.00%", end='\r')

    queryListLength.value = len(queryList)
    progression.value = 0

    iterable = [(x, searchList, nbThresholds, findOnePerSweep, sequenced, useDirectoryName) for x in queryList]
    pool = Pool()
    pool.starmap(job, iterable)
    pool.close()
    pool.join()

    if PERCENTAGE:
        # Line return after percentage print
        print()

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
    parser.add_argument('querydirectorypath')
    parser.add_argument('searchpatternpath')

    printGroup = parser.add_mutually_exclusive_group()
    printGroup.add_argument('-p', '--percentage', action='store_true', help='Enable percentage display')
    printGroup.add_argument('-v', '--verbose', action='store_true', help='Enable verbose display')
    
    args = parser.parse_args()

    PERCENTAGE = args.percentage
    VERBOSE = args.verbose
    queryDirectoryPath = args.querydirectorypath
    resultsDirectoryName = args.resultsname
    searchPatternPath = args.searchpatternpath

    dtw.VERBOSE = VERBOSE
    stats.VERBOSE = VERBOSE

    searchList = sorted(glob.glob(searchPatternPath))
    queryList = []
    for directoryName in next(os.walk(queryDirectoryPath))[1]:
        for fileName in sorted(os.listdir(queryDirectoryPath + directoryName)):
            queryList.append(queryDirectoryPath + directoryName + "/" + fileName)

    resultsPath = RESULTS_ROOT_DIRECTORY + resultsDirectoryName.rstrip('/') + "/"

    print("Running search...")
    AUC, pivot = run(queryList, searchList, nbThresholds=1000, findOnePerSweep=False, sequenced=True, useDirectoryName=True)
    save(AUC, pivot, resultsPath, "test")

    print("Done")

    plt.show()
