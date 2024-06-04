import sys
import os
import random
import yaml
from gen_characters.gen_character import get_character_masks

# Add the path to the directory containing test.py
sys.path.append(os.path.abspath('../../Background_Removal/pytorch-CycleGAN-and-pix2pix'))
import test

def get_random_text():
    first_position = "0123456789"
    second_position = "AHKZP"
    rest_positions = "0123456789"
    
    # Generate a random text according to the specified pattern
    random_text = random.choice(first_position) + random.choice(second_position) + \
                  ''.join(random.choice(rest_positions) for _ in range(3))
    return random_text


def main():
    # Load the configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print("hi... config is set up")

    for i in range(1):
        get_character_masks(get_random_text(), config)

    

if __name__ == "__main__":
    main()