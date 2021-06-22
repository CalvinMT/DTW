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

def buildSplitList(searchFileList, split):
    """
    Based on DyLNet directory structure.
    """
    splitList = []
    for i, searchFile in enumerate(searchFileList):
        splitList.append([])
        transcriptFile = searchFile.rstrip(".wav") + "_VB.ods"
        if not os.path.isfile(transcriptFile):
            if VERBOSE:
                print("ODS file not found: " + transcriptFile)
            continue
        transcript = pd.read_excel(transcriptFile, names=["begin", "end", "text"], engine="odf")
        splittedLength = int(len(transcript) / split)
        for j in range(1, split - 1):
            splitList[i].append(transcript.begin[splittedLength * j])
    return splitList

# XXX - splitList
def buildExpectations(query, searchPatternPath="", searchPathList=None, sweepStep=3, inSequence=False, splitList=[]):
    """
    Based on DyLNet directory structure.
    """
    expectations = []
    queryWord = unidecode(' '.join(re.findall('([A-Za-z]+)', query.split("_")[0])))
    if searchPathList == None:
        searchFileList = sorted(glob.glob(searchPatternPath))
    else:
        searchFileList = searchPathList
    for i, searchFile in enumerate(searchFileList):
        expectations.append([])
        transcriptFile = searchFile.rstrip(".wav") + "_VB.ods"
        if not os.path.isfile(transcriptFile):
            if VERBOSE:
                print("ODS file not found: " + transcriptFile)
            continue
        transcript = pd.read_excel(transcriptFile, names=["begin", "end", "text"], engine="odf")
        for row in transcript.itertuples():
            # XXX
            if row.begin >= splitList[i][-1]:
                break
            if queryWord in unidecode(row.text):
                if inSequence:
                    expectations[i].append([[row.begin, row.end], 1])
                else:
                    for j in range(int(row.begin / sweepStep), int(row.end / sweepStep)):
                        expectations[i].append([j, 1])
            else:
                if inSequence:
                    expectations[i].append([[row.begin, row.end], 0])
                else:
                    for j in range(int(row.begin / sweepStep), int(row.end / sweepStep)):
                        expectations[i].append([j, 0])
    return expectations

def job(query, nbThresholds=1000, oneWord=False, inSequence=True, split=100):
    queryPath = queryDirectoryPath + query

    if split > 1:
        splitList = buildSplitList(searchList, split)
    else:
        splitList = None
    
    # XXX
    splitList = [splitList[0][:1]]

    _, sweepList, _ = dtw.runSearch(queryPath, searchPathList=searchList, splitList=splitList)

    # XXX - splitList
    expectations = buildExpectations(query, searchPathList=searchList, inSequence=True, splitList=splitList)

    if STATS:
        AUC, pivot = stats.computeROCCurve(sweepList, expectations, nbThresholds=nbThresholds, oneWord=oneWord, inSequence=inSequence)
        AUCList.append(AUC)
        pivotList.append(pivot)

    if PERCENTAGE:
        progression.value += 1
        print("%.2f" % (progression.value * 100 / dataLength.value) + "%", end='\r')

def run(data, path, trainingPathList, nbThresholds=1000, oneWord=False):
    """
    TODO
    """
    dataLength.value = len(data)
    progression.value = 0

    pool = Pool()
    # TODO - use starmap to send more arguments
    pool.map(job, data)
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
    queryDirectoryPath = searchDirectoryPath + "Morceaux/"
    # XXX
    #searchList = sorted(glob.glob(searchDirectoryPath + "*R.wav"))
    searchList = [searchDirectoryPath + '21-20181009-070000-1033-4152-03-part02-R.wav']
    # XXX
    #queryList = sorted(os.listdir(queryDirectoryPath))
    queryList = ['arrete-1_21-20181009-070000-1033-4152-03-part02-R_1885512_1889177.wav']

    if STATS and GRAPH:
        figure = plt.figure()

    resultsPath = "results/" + "dtw_dylnet" + "/"

    print("Running search...")
    AUC, pivot = run(queryList, None, searchList)
    save(AUC, pivot, resultsPath, "test")

    print("Done")

    plt.show()