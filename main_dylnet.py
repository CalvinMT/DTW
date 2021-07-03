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

GRAPH = False
PERCENTAGE = False
STATS = False
VERBOSE = False

# TODO - clean up
manager = Manager()
dataLength = manager.Value('i', 0)
progression = manager.Value('i', 0)
AUCList = manager.list()
pivotList = manager.list()

def buildQuerySplitList(queryList):
    """
    Based on DyLNet directory structure.
    """
    querySplitList = []
    for query in queryList:
        tmp = query.split(".")[0]
        begin = int(tmp.split("_")[-2])
        end = int(tmp.split("_")[-1])
        searchFileName = tmp.split("_")[1]
        querySplitList.append([begin, end, searchFileName])
    return querySplitList

def buildExpectations(queryPath, searchPatternPath="", searchPathList=None, sweepStep=3, inSequence=False, useDirectoryName=False):
    """
    Based on DyLNet directory structure.
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
            if VERBOSE:
                print("ODS file not found: " + transcriptFile)
            continue
        transcript = pd.read_excel(transcriptFile, names=["begin", "end", "text"], engine="odf")
        for row in transcript.itertuples():
            if row.begin != begin or row.end != end:
                continue
            if queryWord in unidecode(row.text):
                if inSequence:
                    expectations[i].append([[0, row.end - row.begin], 1])
                else:
                    for j in range(int(row.begin / sweepStep), int(row.end / sweepStep)):
                        expectations[i].append([j, 1])
            else:
                if inSequence:
                    expectations[i].append([[0, row.end - row.begin], 0])
                else:
                    for j in range(int(row.begin / sweepStep), int(row.end / sweepStep)):
                        expectations[i].append([j, 0])
    return expectations

def job(queryPath, searchList, nbThresholds=1000, oneWord=False, inSequence=False, useDirectoryName=False):
    # Remove search file from which query comes from
    searchListWithoutQuery = []
    query = queryPath.split("/")[-1]
    for search in searchList:
        if query.rstrip(".wav").split("_", 1)[1] not in search:
            searchListWithoutQuery.append(search)

    _, sweepList, _ = dtw.runSearch(queryPath, searchPathList=searchListWithoutQuery)

    expectations = buildExpectations(queryPath, searchPathList=searchListWithoutQuery, inSequence=inSequence, useDirectoryName=useDirectoryName)

    if STATS:
        AUC, pivot = stats.computeROCCurve(sweepList, expectations, nbThresholds=nbThresholds, oneWord=oneWord, inSequence=inSequence)
        AUCList.append(AUC)
        pivotList.append(pivot)

    if PERCENTAGE:
        progression.value += 1
        print("%.2f" % (progression.value * 100 / dataLength.value) + "%", end='\r')

def run(data, searchList, nbThresholds=1000, oneWord=False, inSequence=False, useDirectoryName=False):
    """
    TODO
    """
    dataLength.value = len(data)
    progression.value = 0

    iterable = [(x, searchList, nbThresholds, oneWord, inSequence, useDirectoryName) for x in data]
    pool = Pool()
    pool.starmap(job, iterable)
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
    parser.add_argument('path')

    printGroup = parser.add_mutually_exclusive_group()
    printGroup.add_argument('-p', '--percentage', action='store_true', help='Enable percentage display')
    printGroup.add_argument('-v', '--verbose', action='store_true', help='Enable verbose display')
    
    args = parser.parse_args()

    GRAPH = args.graph
    PERCENTAGE = args.percentage
    STATS = args.stats
    VERBOSE = args.verbose
    path = args.path

    dtw.PERCENTAGE = PERCENTAGE
    dtw.VERBOSE = VERBOSE
    stats.VERBOSE = VERBOSE

    searchDirectoryPath = path.rstrip('/') + "/"
    queryDirectoryPath = searchDirectoryPath + "Segments/"
    searchList = sorted(glob.glob(searchDirectoryPath + "Morceaux/" + "*R_*_*.wav"))
    #queryList = sorted(os.listdir(queryDirectoryPath))
    queryList = []
    for directoryName in next(os.walk(queryDirectoryPath))[1]:
        for fileName in sorted(os.listdir(queryDirectoryPath + directoryName)):
            queryList.append(queryDirectoryPath + directoryName + "/" + fileName)

    #querySplitList = buildQuerySplitList(queryList)

    if STATS and GRAPH:
        figure = plt.figure()

    resultsPath = "results/" + "dtw_dylnet" + "/"

    print("Running search...")
    AUC, pivot = run(queryList, searchList, nbThresholds=1000, oneWord=False, inSequence=True, useDirectoryName=True)
    save(AUC, pivot, resultsPath, "test")

    print("Done")

    plt.show()
