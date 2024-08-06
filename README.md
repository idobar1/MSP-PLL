# How to run

## 1. Prerequisites
pip install pydub

pip install strenum

pip install numpy

pip install scipy

pip install matplotlib


## 2. Comment and Uncomment Relevant Parts
Open main.py, uncomment uncomment the relevant section, for example:

between: ### Input 4: Square Wave, dc = 0.1 ###

and: ### <<< End of Input 4 >>>

The rest of the sections: comment out

### Choose your own song
1. Uncomment the first section.

2. Put the song file in the main directory

3. Go to config.py and inside FileConfig change the name and format accordingly.

    You can also use DebugConfig to configure cut_start_sec for setting the beginning of the processed signal, and cut_len_sec for its duration (in seconds). Please choose a length bigger than 0 for the processing signal.

4. Run `python main.py`

5. After finished, the file 'Music and Metronome.wav' will be created. Run it to listen to the song with the metronome sound.

## 3. Choose PLL parameters
At the beginning of the chosen section, configure the MathConfig instances as you wish
loop_gain, filt_type (can be either FiltType.GAIN or FiltType.MA), loop_filter_mem (memory for moving average), VCO_gain and f0.

You can add more instances to math_config_list if you wish to run more examples.

## 4. Run
run `python main.py`

Guide to download & install ffmpeg (relevant for python libraries dealing with files including mp3 which belongs to MPEG):
https://phoenixnap.com/kb/ffmpeg-windows


Documentation of pydub:
https://github.com/jiaaro/pydub


Nice tutorial for working with GitHub in VSCODE:
https://www.youtube.com/watch?v=S7TbHDN8EXA&list=PLpPVLI0A0OkLBWbcctmGxxF6VHWSQw1hi&index=5

