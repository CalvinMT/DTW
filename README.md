# Dynamic Time Warping

## Examples

### General

`python ./main.py wav/fr/query_fr.wav 'wav/fr/search?_fr.wav' -vgs`

or

`cat speech_command_v0.02.txt | xargs python ./main.py -vgs`

`cat dylnet.txt | xargs python ./main.py -g`



### Speech Command v0.02

| Option | Description |
|--------|-------------|
| -h     | Show help. |
| -n     | Number of thresholds to build ROC curve. |
| -p     | Enable percentage display. |
| -r     | Name of the directory containing the results. |
| -t     | Enable trimming of test, validation and training lists to the given percentage. |
| -v     | Enable verbose display |

`python ./main_speech_command_v0.02.py -p -t=0.05 -r='dtw_sc2_0.05' <data_path>`



### DyLNet

| Option | Description |
|--------|-------------|
| -h     | Show help. |
| -n     | Number of thresholds to build ROC curve. |
| -p     | Enable percentage display. |
| -r     | Name of the directory containing the results. |
| -v     | Enable verbose display. |

`python ./main_dylnet.py -p -r='dtw_dylnet_test' <queries_path> <search_pattern_path>`



### Results

`python ./main_results.py -sga results/<folder>/`
