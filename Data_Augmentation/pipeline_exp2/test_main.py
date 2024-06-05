import sys
import os
import random
import yaml
from gen_characters.gen_character import get_character_masks

def main():
    # Load the configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print("hi... config is set up")

    

    

if __name__ == "__main__":
    main()