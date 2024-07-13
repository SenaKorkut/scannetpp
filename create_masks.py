import pickle
import pprint
import torch
import numpy as np
from PIL import Image
import json
import argparse
import os 


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
    #parser.add_argument('file3', type=str, help='Path to the third file')
    
    args = parser.parse_args()

    pth_filepath = args.pth_file_dir
    pkl_filepath = args.rasterized_file_dir

    # Get top 10 instance labels from metadata file 
    top_labels_dir = '/storage/user/yez/scannet++/metadata/semantic_benchmark/top100_instance.txt'
    top_labels = get_top_labels(num_labels=10, file_path=top_labels_dir)

    scenes = list_files(pth_filepath)

    for scene in scenes:

        scene_path = os.path.join(pth_filepath, scene)
        # Read .pth file
        pth_data = read_pth_file(pth_filepath)
        if pth_data is not None:
            print("Successfully read the .pth file")
        
        # Get the scene ID from the .pth file
        scene_id = pth_data['scene_id']
        # Get the segments_anno.json file for the scene
        segments_anno = get_segments_anno_file(scene_id)
        # Extract segGroups
        seg_groups = segments_anno.get('segGroups', [])
        segmentation_path = os.path.join(pkl_filepath, scene)

        for image_name in os.listdir(segmentation_path):

            pkl_filepath = os.path.join(segmentation_path, image_name)
            # Read .pkl file
            pkl_data = read_pkl_file(pkl_filepath)
            if pkl_data is not None:
                print("Successfully read the .pkl file:")

            # Filter segGroups based on labels to be used 
            for label in top_labels:

                label_ids = [group['id'] for group in seg_groups if group.get('label') == label]
                print("Label IDs:", label_ids)

                for label_id in label_ids:
                    # Create the mask
                    mask = create_mask(pkl_data, label_id)
                    # Save the mask as a PNG file
                    output_filepath = f'mask_{label_id}.png'
                    save_mask_as_png(mask, output_filepath)
                    print(f"Mask saved as {output_filepath}")