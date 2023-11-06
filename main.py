import glob
import pretty_midi
import pygame
import pathlib

from IPython import display

data_dir = pathlib.Path('music_data')
filenames = glob.glob(str(data_dir/'*.mid*'))
print(filenames)
sample_file = filenames[1]
print(sample_file)
pm = pretty_midi.PrettyMIDI(sample_file)