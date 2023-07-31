import os
from PIL import Image
import random


SMALL_DATASET = True
SMALL_DATASET_SIZE = 144


# Function to split the image into 4x4 pieces and save
def split_and_save_images(file_list, dest_subdir, src_dir, dest_dir, size=400):
    for filename in file_list:
        # Open the image file
        img = Image.open(src_dir + filename)
        width, height = img.size

        # Split the image
        for i in range(2):  # Only two iterations for y-direction
            for j in range(2):  # Only two iterations for x-direction
                # Calculate the boundaries for the chunk
                start_x = j * 400
                start_y = i * 400
                end_x = (j+1) * 400
                end_y = (i+1) * 400

                # Crop the image
                cropped_img = img.crop((start_x, start_y, end_x, end_y))
                if '.jpg' in filename:
                    cropped_img = cropped_img.convert('RGBA')
                if 'mask' in filename:
                    cropped_img = cropped_img.convert('L')
                # Save the cropped image
                cropped_img.save(dest_dir + dest_subdir + filename.split('.')[0] + f'_{i*2 + j}.' + 'png')


def main():
    src_dir = "Datasets/DeepGlobe/train/"
    dest_dir = "my_dataset_small_from_deepglobe_plus_ethz/training/"

    # Create the directories if not already exist
    os.makedirs(dest_dir + 'images', exist_ok=True)
    os.makedirs(dest_dir + 'groundtruth', exist_ok=True)

    # List all the files in the source directory
    files = os.listdir(src_dir)
    files = [file.rsplit('_')[0] for file in files if file.endswith(".png")]
    if SMALL_DATASET:
        files = random.sample(files, k=SMALL_DATASET_SIZE)

    # Filter for the satellite images and the mask images
    sat_files = [f'{file}_sat.jpg' for file in files]
    mask_files = [f'{file}_mask.png' for file in files]

    # Split and save the satellite images
    split_and_save_images(sat_files, 'images/', src_dir, dest_dir)

    # Split and save the mask images
    split_and_save_images(mask_files, 'groundtruth/', src_dir, dest_dir)

    print("Finished!")


if __name__ == '__main__':
    main()
