import numpy as np
from PIL import Image
import json
import argparse
import os 


def get_camera_pose_as_npy(img_name):
    for img in transforms_json["frames"]:
        if img["file_path"] == img_name:
            transformation_matrix = img["transform_matrix"]
            transformation_matrix = np.array(transformation_matrix)
            return transformation_matrix
    print("Image not found.")
    return None

def load_json_file(json_filepath):
    """Load and return the contents of a JSON file."""
    try:
        with open(json_filepath, "r") as json_file:
            data = json.load(json_file)
        return data
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return None
    
def open_image_as_array(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Convert the image to RGB format (in case it's not)
        img = img.convert("RGB")
        # Convert the image to a NumPy array
        image_array = np.array(img)

    return image_array

def list_files(directory):
    """List all files in the given directory."""
    try:
        # Get a list of all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return files
    except Exception as e:
        print(f"Error accessing directory {directory}: {e}")
        return []
    
def list_folders(directory):
    """List all folders in the given directory."""
    try:
        # Get a list of all folders in the directory
        folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
        return folders
    except Exception as e:
        print(f"Error accessing directory {directory}: {e}")
        return []

if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Create test data based on given masks.')
    parser.add_argument('masks_dir', type=str, help='Path to the masks folder')
    parser.add_argument('output_dir', type=str, help='Path to the output folder directory')
    
    args = parser.parse_args()
    masks_filepath = args.masks_dir
    output_dir = args.output_dir

    dataset_dir = "/storage/user/yez/scannet++/data"

    print(f"Creating test data from masks in {masks_filepath}")

    #Get the list of scenes from masks_filepath directory
    scenes = list_folders(masks_filepath)

    print(f"Found {len(scenes)} scenes in the directory.")

    for scene in scenes:

        print(f"Processing scene: {scene}")
        # Get the scene masks from masks_filepath
        mask_scene_path = os.path.join(masks_filepath, scene)
        # Get the scene information from original scannetpp dataset
        dataset_scene_path = os.path.join(dataset_dir, scene)
        # Get resized images path from original scannetpp dataset
        dataset_scene_images_path = os.path.join(dataset_scene_path, "dslr", "resized_images")
        # Get camera transformation json file path from original scannetpp dataset
        camera_pose_transform_json_path = os.path.join(
            dataset_scene_path, "dslr", "nerfstudio", "transforms.json"
        )
        # Read json file
        transforms_json = load_json_file(camera_pose_transform_json_path)
        
        for mask_name in os.listdir(mask_scene_path):

            # Get mask path 
            mask_path = os.path.join(mask_scene_path, mask_name)
            # Split the mask name to get image_name 
            image_name = mask_name.split("_")[0]
            # Get the folder name for the data as  {scene}_{label}_{instance_id}
            out_folder_name = scene + "_" + mask_name.split("_")[1].split(".")[0] + "_" + mask_name.split("_")[2].split(".")[0]
            # Get the original image from scannetpp dataset
            dslr_image_path = os.path.join(dataset_scene_images_path, f"{image_name}.JPG")

            # Extract the object from the DSLR image using the mask
            dslr_image_array = open_image_as_array(dslr_image_path)
            mask_array = open_image_as_array(mask_path)
            extracted_object = np.array(dslr_image_array.copy())
            extracted_object[mask_array != 255] = 0

            # Save the extracted object as a PNG file
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(f"{output_dir}/{out_folder_name}"):
                os.makedirs(f"{output_dir}/{out_folder_name}")
            output_filepath_extracted = (f"{output_dir}/{out_folder_name}/{image_name}")
            Image.fromarray(extracted_object).save(output_filepath_extracted + ".png")
            print(f"Extracted object saved as {output_filepath_extracted}")

            # Save the corresponding camera pose as a .npy file
            camera_pose = get_camera_pose_as_npy(f'{image_name}.JPG')
            np.save(output_filepath_extracted + ".npy", camera_pose)
            print(f"Matrix saved successfully at {output_filepath_extracted}")