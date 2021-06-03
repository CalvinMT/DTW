import dtw
import stats

import glob
import argparse
import matplotlib.pyplot as plt

GRAPH = False
PERCENTAGE = False
STATS = False
VERBOSE = False

def buildExpectations(queryPath, searchPatternPath):
    """
    Based on SpeechCommand_v0.02 directory structure.
    """
    expectations = []
    currentDirectory = ""
    queryFilename = queryPath.split("/")[-1]
    queryDirectory = queryPath.split("/")[-2]
    queryCode = queryFilename.split("_")[0]
    searchFileList = sorted(glob.glob(searchPatternPath))
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

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Dynamic Time Warping')
    parser.add_argument('-g', '--graph', action='store_true', help='Enable graph display')
    parser.add_argument('-s', '--stats', action='store_true', help='Enable statistics display')
    parser.add_argument('search_pattern_path')

    printGroup = parser.add_mutually_exclusive_group()
    printGroup.add_argument('-p', '--percentage', action='store_true', help='Enable percentage display')
    printGroup.add_argument('-v', '--verbose', action='store_true', help='Enable verbose display')
    
    args = parser.parse_args()

    GRAPH = args.graph
    PERCENTAGE = args.percentage
    STATS = args.stats
    VERBOSE = args.verbose
    searchPatternPath = args.search_pattern_path

    dtw.VERBOSE = VERBOSE
    stats.VERBOSE = VERBOSE

    queryPatternPath = searchPatternPath.rsplit('/', 1)[0]
    queryPatternPath += "/a331d9cb_nohash_*.wav"
    queryPathList = sorted(glob.glob(queryPatternPath))

    rows = []
    columns = []
    currentDirectory = ""
    for queryPath in queryPathList:
        directory = queryPath.split("/")[-2]
        if directory != currentDirectory:
            rows.append(1)
            columns.append(1)
            currentDirectory = directory
        else:
            columns[-1] += 1
    plotpos = [len(rows), max(columns), 0]

    if STATS and GRAPH:
        figure = plt.figure()

    AUCList = []
    pivotList = []
    r = 0
    c = 0
    for i, queryPath in enumerate(queryPathList):
        title = queryPath.split("/")[-2] + "/" + queryPath.split("/")[-1]

        labels, sweepList, bestList = dtw.runSearch(queryPath, searchPatternPath)

        expectations = buildExpectations(queryPath, searchPatternPath)

        if STATS:
            AUC, pivot = stats.computeROCCurve(sweepList, expectations, nbThresholds=1000, oneWord=True)
            AUCList.append(AUC)
            pivotList.append(pivot)
            if GRAPH:
                plotpos[2] += 1
                c += 1
                if c > columns[r]:
                    plotpos[2] += max(columns) - columns[r]
                    r += 1
                    c = 1
                stats.showROCCurve(AUC, pivot, figure=figure, plotpos=plotpos, title=title)

        if PERCENTAGE:
            print("%.2f" % (i * 100 / len(queryPathList)) + "%", end='\r')

    # Mean ROC
    assert(len(AUCList) == len(pivotList))
    sumAUC = AUCList[0]
    sumPivot = pivotList[0]
    for i in range(1, len(AUCList)):
        sumAUC += AUCList[i]
        sumPivot += pivotList[i]
    meanAUC = sumAUC / len(AUCList)
    meanPivot = sumPivot / len(pivotList)
    stats.showROCCurve(meanAUC, meanPivot, title="Mean ROC")

    plt.show()
