import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from glob import glob
import os
import random
import shutil
from copy import deepcopy
from tqdm.notebook import tqdm

import json
from create_mask import create_character_mask

def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # create_character_mask
    possible_characters = "AHKZP0123456789"
    for i in tqdm(range(1000)):
        create_character_mask(random.choice(possible_characters), i, config)

if __name__ == '__main__':
    print("Hi")
    main()

