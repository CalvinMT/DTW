import matplotlib.pyplot as plt
import glob
import numpy as np
import scipy.io.wavfile as wav
import operator
from speech_dtw import qbe
from python_speech_features import mfcc
from python_speech_features import delta

PATH = "wav/fr/"
QUERY_FILE_NAME = "query_fr.wav"
SEARCH_FILE_NAME = "search?_fr.wav"

def GetMFCC(signal, rate):
    mel = mfcc(signal, rate) # x: time, y: coefficients
    delta1 = delta(mel, 2) # speed
    delta2 = delta(delta1, 2) # acceleration
    features = np.hstack([mel, delta1, delta2])
    cmvn = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    return cmvn

(rate, query) = wav.read(PATH + QUERY_FILE_NAME)
"""plt.plot(query)
plt.show()"""

queryMfcc = GetMFCC(query, rate)
"""plt.imshow(queryMfcc.T, interpolation="nearest")
plt.show()"""

labels = []
sweepList = []
for file_name in sorted(glob.glob(PATH + SEARCH_FILE_NAME)):
    labels.append(file_name.split("/")[-1])
    (rate, search) = wav.read(file_name)
    searchMfcc = GetMFCC(search, rate)
    sweepList.append(qbe.dtw_sweep(queryMfcc, searchMfcc, 3)) # x: time/3, y: cost

axes = plt.gca()
axes.set_ylim([0.2, 0.6])
scoreList = [[0, 0] for i in range(len(sweepList))] # 0: index; 1: minimum
for i, sweep in enumerate(sweepList):
    color = next(axes._get_lines.prop_cycler)['color']
    plt.plot(sweep, ':', color=color, zorder=i)
    # plot minimum point
    scoreList[i][0], scoreList[i][1] = min(enumerate(sweep), key=operator.itemgetter(1))

color = axes.set_prop_cycle(None)
for i, score in enumerate(scoreList):
    color = next(axes._get_lines.prop_cycler)['color']
    plt.plot(score[0], score[1], 'o', color=color, zorder=i+len(sweepList)-1)

axes.set_xlabel('Time', fontsize='x-large')
axes.set_ylabel('Score', fontsize='x-large')
axes.legend(labels)
plt.show()
