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
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Set score threshold')
    parser.add_argument('query_path')
    parser.add_argument('search_pattern_path')

    printGroup = parser.add_mutually_exclusive_group()
    printGroup.add_argument('-p', '--percentage', action='store_true', help='Enable percentage display')
    printGroup.add_argument('-v', '--verbose', action='store_true', help='Enable verbose display')

    args = parser.parse_args()

    GRAPH = args.graph
    PERCENTAGE = args.percentage
    STATS = args.stats
    threshold = args.threshold
    VERBOSE = args.verbose
    queryPath = args.query_path
    searchPatternPath = args.search_pattern_path

    dtw.VERBOSE = VERBOSE
    stats.VERBOSE = VERBOSE

    labels, sweepList, bestList = dtw.runSearch(queryPath, searchPatternPath)

    if GRAPH:
        dtw.showSweeps(labels, sweepList, bestList)

    expectations = buildExpectations(queryPath, searchPatternPath)

    if STATS:
        #stats.printStatistics(results, expectations)
        if GRAPH:
            AUC, pivot = stats.computeROCCurve(sweepList, expectations, nbThresholds=1000, oneWord=True)
            stats.showROCCurve(AUC, pivot)

    plt.show()
