import os
from glob import glob
import shutil
import subprocess

# relative import
from .create_mask import create_character_mask
from .select_attributes import get_random_mask_size, get_random_font, get_random_model_checkpoints


def get_character_masks(text, config):
    # set up attribute parameters
    # get fixed random mask_size & fixed randon font - need 5 characters of the text have the same setting
    characters = list(text)
    font = get_random_font(config['fonts_dir'])
    mask_size = get_random_mask_size(config['candidate_mask_sizes'])

    # create each character mask & fix the same font for each character in the text
    for i in range(len(characters)):
        create_character_mask(char=characters[i], font=font, mask_size=mask_size, pic_index=i, config=config)
    
    # randonly select a model checkpoint
    exp_name = get_random_model_checkpoints(config['exp_names'])

    # run the test script to get stylized result images
    run_test_script(config, exp_name, mask_size, len(characters))

    # rename for unified/single one result output - config['generated_chars_image_dir']
    rename_result_dir(config, exp_name)

    # remove existing used images in dataset - testB
    for i in glob(os.path.join(config["mask_created_save_dir"], "*")):
        os.remove(i)



def rename_result_dir(config, exp_name):
    work_dir = "C:/Users/User/Desktop/Pice/Work/ConnectedTech/UTAC_OCR/Data_Augmentation/pipeline_exp2"
    generated_output_dir = f"{work_dir}/{config['generated_chars_dir']}/{exp_name}"
    renamed_dir = f"{work_dir}/{config['generated_chars_dir']}/results"
    os.rename(generated_output_dir, renamed_dir)



def run_test_script(config, exp_name, mask_size, num_characters):
    # Define the path to the test.py script
    test_script_path = os.path.abspath('../../Background_Removal/pytorch-CycleGAN-and-pix2pix/test.py')

    # Define the directory of the test.py script
    script_dir = os.path.dirname(test_script_path)

    # to access our working directory
    working_dir = "../../Data_Augmentation/pipeline_exp2"

    # Run the test.py script using subprocess
    command = [
        'python', test_script_path,
        '--dataroot', os.path.join(working_dir, config['mask_created_dir']),
        '--direction', 'BtoA',
        '--checkpoints_dir', os.path.join(working_dir, config['checkpoints_dir']),
        '--name', exp_name,
        '--results_dir', os.path.join(working_dir, config['generated_chars_dir']),
        '--preprocess', 'scale_width_and_crop',
        '--load_size', str(mask_size),
        '--crop_size', str(mask_size),
        '--num_test', str(num_characters),
        '--model', 'test',
        '--no_dropout' # >>> must add no dropout !!!!!
    ]

    result = subprocess.run(command, cwd=script_dir, capture_output=True, text=True)

    # Print the output and error (if any)
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)




# Old version - to use test.py from CycleGan
# os.system(f"python test.py --dataroot {config['mask_created_dir']} --direction BtoA --checkpoints_dir \
#         {config['checkpoints_dir']} --name {exp_name} --results_dir {config['generated_chars_dir']} \
#         --preprocess scale_width_and_crop --load_size {mask_size} --crop_size {mask_size} \
#         --num_test {len(characters)} --model test --no_dropout")