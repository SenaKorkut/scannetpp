import pickle
import pprint
import torch
import numpy as np
from PIL import Image
import json
import argparse
import os 

'''
def get_camera_pose_as_npy(img_name):
    for img in transforms_json["frames"]:
        if img["file_path"] == img_name:
            transformation_matrix = img["transform_matrix"]
            transformation_matrix = np.array(transformation_matrix)
            return transformation_matrix
    print("Image not found.")
    return None
'''


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

def read_pkl_file(pkl_filepath):
    """Read and return the contents of a .pkl file."""
    try:
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        return data
    except Exception as e:
        print(f"An error occurred while reading the .pkl file: {e}")
        return None

def read_pth_file(pth_filepath):
    """Read and return the contents of a .pth file."""
    try:
        data = torch.load(pth_filepath)
        return data
    except Exception as e:
        print(f"An error occurred while reading the .pth file: {e}")
        return None

def create_mask(instance_ids, target_id):
    # Create a mask where the instance ID is equal to target_id
    mask = np.zeros_like(instance_ids, dtype=np.uint8)
    mask[instance_ids == target_id] = 255  # Set target ID to white (255)
    return mask

def save_mask_as_png(mask, filepath):
    # Convert the mask to a PIL Image
    mask_image = Image.fromarray(mask)
    # Save the image as a PNG file
    mask_image.save(filepath)

def get_segments_anno_file(scene_id):
    dataset_dir = '/storage/user/yez/scannet++/data'
    scene_dir = f'{dataset_dir}/{scene_id}'
    segments_anno_json_dir = f'{scene_dir}/scans/segments_anno.json'

    # Read the segments_anno.json file as a json file 
    with open(segments_anno_json_dir, 'r') as f:
        segments_anno = json.load(f)
    
    return segments_anno

def get_top_labels(num_labels, file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()
        # Get the top num_labels lines
        top_lines = lines[:num_labels]
        
    return top_lines

def list_files(directory):
    """List all files in the given directory."""
    try:
        # Get a list of all files in the directory
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return files
    except Exception as e:
        print(f"Error accessing directory {directory}: {e}")
        return []

if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Create masks based on given path files.')
    parser.add_argument('pth_file_dir', type=str, help='Path to the .pth files created by prepare_training_data.py')
    parser.add_argument('rasterized_file_dir', type=str, help='Path to the .pkl files created by rasterize_semantics_2d.py')
    parser.add_argument('masks_output_dir', type=str, help='Path to the output masks file')
    
    args = parser.parse_args()

    scene_data_filepath = args.pth_file_dir
    mask_data_filepath = args.rasterized_file_dir
    mask_output_filepath = args.masks_output_dir

    # Get top 10 instance labels from metadata file 
    top_labels_dir = '/storage/user/yez/scannet++/metadata/semantic_benchmark/top100_instance.txt'
    top_labels = get_top_labels(num_labels=10, file_path=top_labels_dir)
    top_labels = [s.strip() for s in top_labels]
    print("Top Labels:", top_labels)

    scenes = list_files(scene_data_filepath)

    for scene in scenes:

        scene_path = os.path.join(scene_data_filepath, scene)
        # Read .pth file
        scene_data = read_pth_file(scene_path)
        if scene_data is not None:
            print("Successfully read the scene file")

        '''dataset_scene_path = os.path.join(dataset_dir, scene.replace(".pth", ""))
        dslr_scene_path = os.path.join(dataset_scene_path, "dslr", "resized_images")
        camera_pose_transform_json_path = os.path.join(
            dataset_scene_path, "dslr", "nerfstudio", "transforms.json"
        )

        transforms_json = load_json_file(camera_pose_transform_json_path)

        '''
        # Get the scene ID from the .pth file
        scene_id = scene_data['scene_id']
        # Get the segments_anno.json file for the scene
        segments_anno = get_segments_anno_file(scene_id)
        # Extract segGroups
        seg_groups = segments_anno.get('segGroups', [])
        array_path = os.path.join(mask_data_filepath, scene_id)

        print("mask_data_filepath", mask_data_filepath)
        print("scene", scene_id)
        print("simages_path", array_path)

        for image_name in os.listdir(array_path):

            image_filepath = os.path.join(array_path, image_name)
            # Read .pkl file
            array_data = read_pth_file(image_filepath)
            if array_data is not None:
                print("Successfully read the array")
            
            print("image_name", image_name)

            #dslr_image = os.path.join(dslr_scene_path, imgae_JPG)

            #Get instance ids that are visible in the current image
            visible_instance_ids = np.unique(array_data)

            # Filter segGroups based on labels to be used 
            for label in top_labels:

                #Get available instance ids from the scene based on a specific label
                instance_ids = [group['id'] for group in seg_groups if group.get('label') == label]
                print("Instance IDs:", instance_ids)

                # Create a new list with values present in both lists
                intersection = [value for value in visible_instance_ids if value in instance_ids]

                # Loop through the intersection list
                for instance_id in intersection:
                    # Create the mask
                    mask = create_mask(array_data, instance_id)
                    # Save the mask as a PNG file

                    new_filepath = os.path.join(mask_output_filepath, scene_id) #Create scene folder for masks

                    if not os.path.exists(new_filepath):
                        os.makedirs(new_filepath)

                    # Extract the image name without the extension
                    img_name_wo_ext = os.path.splitext(os.path.basename(image_name))[0]

                    # Create mask name
                    label = label.replace(" ", "-")
                    mask_name = f"{img_name_wo_ext}_{label}_{instance_id}.png"
                    mask_save_path = os.path.join(new_filepath, mask_name)

                    #Save mask as PNG
                    save_mask_as_png(mask, mask_save_path)

                    '''
                    output_dir_extracted = (
                    f"extracted/{image_name.replace('_inst.pkl', '')}"
                )
                    
                    os.makedirs(output_dir_mask, exist_ok=True)
                    os.makedirs(output_dir_extracted, exist_ok=True)

                    save_mask_as_png(mask, output_filepath)
                    print(f"Mask saved as {output_filepath}")

                    # Extract the object from the DSLR image using the mask
                    dslr_image_array = open_image_as_array(dslr_image)
                    extracted_object = np.array(dslr_image_array.copy())
                    extracted_object[mask != 255] = 0

                    # Save the extracted object as a PNG file
                    output_filepath_extracted = (
                        f"{output_dir_extracted}/extracted_{label_id}"
                    )
                    Image.fromarray(extracted_object).save(
                        output_filepath_extracted + ".png"
                    )
                    print(f"Extracted object saved as {output_filepath_extracted}")

                    # Save the camera pose
                    camera_pose = get_camera_pose_as_npy(imgae_JPG)
                    np.save(output_dir_extracted + ".npy", camera_pose)
                    print(f"Matrix saved successfully at {output_dir_extracted}")'''