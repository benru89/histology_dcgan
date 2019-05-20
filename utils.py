import json
import re
from random import sample
from shutil import copyfile
import os

def change_active_config(active_config):
    with open("config.json", "r") as conffile:
        config = json.load(conffile)

    config["active_config"] = active_config

    with open("config.json", "w") as conffile:
        json.dump(config, conffile)


def copy_n_rand_patches(src_dir, dest, n):    
    for i, file in enumerate(sample(os.listdir(src_dir), n)):
        copyfile(src_dir+file, dest+file)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
