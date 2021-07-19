# Dynamic Time Warping

## Examples

### General

`python ./main.py wav/fr/query_fr.wav 'wav/fr/search?_fr.wav' -vgs`

or

`cat speech_command_v0.02.txt | xargs python ./main.py -vgs`

`cat dylnet.txt | xargs python ./main.py -g`



### Speech Command v0.02

`python ./main_speech_command_v0.02.py -p -t=0.05 -r='dtw_sc2_0.05' <data_path>`



### DyLNet

`python ./main_dylnet.py -p -r='dtw_dylnet_test' <queries_path> <search_pattern_path>`



### Results

`python ./main_results.py -sga results/<folder>/`
