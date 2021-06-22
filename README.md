# Dynamic Time Warping

## Examples

### General

`python ./main.py wav/fr/query_fr.wav 'wav/fr/search?_fr.wav' -vgs`

or

`cat speech_command_v0.02.txt | xargs python ./main.py -vgs`

`cat dylnet.txt | xargs python ./main.py -g`



### Speech Command v0.02

`python ./main_speech_command_v0.02.py '../../Datasets/speech_commands_v0.02/' -ps`

old (`python ./main_speech_command_v0.02.py '../../Datasets/speech_commands_v0.02/*/a331d9cb_nohash_*.wav' -ps`)



### DyLNet

`python ./main_dylnet.py -ps ../../Datasets/dylnet/Selection_morceaux/`



### Results

`python ./main_results.py -sgt results/<folder>/`
