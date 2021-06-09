import stats

import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='ROC plotter')
    parser.add_argument('path', type=str, help='Directory path to pivots and AUC files')
    parser.add_argument('-g', '--graph', action='store_true', help='Enable show plotted image')
    parser.add_argument('-s', '--save', action='store_true', help='Enable save plotted image')
    parser.add_argument('-a', '--all', action='store_true', help='Enable all sets')
    parser.add_argument('-t', '--test', action='store_true', help='Enable test set')
    parser.add_argument('-v', '--validation', action='store_true', help='Enable validation set')
    args = parser.parse_args()

    directoryPath = str(args.path).rstrip('/') + "/"
    GRAPH = args.graph
    SAVE = args.save
    ALL_SETS = args.all
    TEST_SET = args.test
    VALIDATION_SET = args.validation

    sets = []
    if ALL_SETS or TEST_SET:
        sets.append("test")
    if ALL_SETS or VALIDATION_SET:
        sets.append("validation")

    for setName in sets:
        pivotPath = directoryPath + setName + "_pivots.csv"
        aucPath = directoryPath + setName + "_auc.txt"

        directoryName = pivotPath.split('/')[-2]
        imageName = directoryName + "_" + setName

        pivot = pd.read_csv(pivotPath)
        
        with open(aucPath, 'r') as f:
            auc = float(next(f))

        stats.showROCCurve(auc, pivot, title=imageName)

        if SAVE:
            plt.savefig(directoryPath + imageName + ".png")

    if GRAPH:
        plt.show()
