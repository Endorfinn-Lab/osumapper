from audio_tools import *
from os_tools import *
import os, re, time

mapdata_path = "mapdata/"

def step4_read_new_map(file_path, divisor = 4):
    fix_path()
    test_process_path("node")
    test_process_path("ffmpeg", "-version")
    test_node_modules()
    start = time.time()
    read_and_save_osu_tester_file(file_path.strip(), filename="mapthis", divisor=divisor)
    end = time.time()