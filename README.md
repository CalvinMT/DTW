GENERAL

python ./main.py wav/fr/query_fr.wav 'wav/fr/search?_fr.wav' -vgs

OR

cat speech_command_v0.02.txt | xargs python ./main.py -vgs



SPEECH COMMAND V0.02

python ./main_speech_command_v0.02.py '../../Datasets/speech_commands_v0.02/' -ps

old (python ./main_speech_command_v0.02.py '../../Datasets/speech_commands_v0.02/*/a331d9cb_nohash_*.wav' -ps)