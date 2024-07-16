import pickle
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

    if os.path.exists(segments_anno_json_dir):

        # Read the segments_anno.json file as a json file 
        with open(segments_anno_json_dir, 'r') as f:
            segments_anno = json.load(f)
    
        return segments_anno
    else:
        print("Segments_anno.json file not found.")
        return None

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
    
def count_pixels(instance_id, mask):
    return np.sum(mask == instance_id)

if __name__ == "__main__":

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Create masks based on given path files.')
    parser.add_argument('pth_file_dir', type=str, help='Path to the .pth files created by prepare_training_data.py')
    parser.add_argument('rasterized_file_dir', type=str, help='Path to the .pkl files created by rasterize_semantics_2d.py')
    parser.add_argument('masks_output_dir', type=str, help='Path to the output masks file')
    parser.add_argument('--difficult_labels', action='store_true', help='Set to true if you want to get masks for difficult labels')
    parser.add_argument('--no_difficult_labels', action='store_false', dest='difficult_labels', help='Set to false if you do not want to get masks for difficult labels')

    
    args = parser.parse_args()

    scene_data_filepath = args.pth_file_dir
    mask_data_filepath = args.rasterized_file_dir
    mask_output_filepath = args.masks_output_dir
    difficult_labels = args.difficult_labels


    # Get top 10 instance labels from metadata file 
    #top_labels_dir = '/storage/user/yez/scannet++/metadata/semantic_benchmark/top100_instance.txt'
    #top_labels = get_top_labels(num_labels=2, file_path=top_labels_dir)
    #top_labels = [s.strip() for s in top_labels]
    print(f"Difficult labels flag: {difficult_labels}")  # Debug statement

    if difficult_labels:
        top_labels = ['blinds', 'curtain', 'blanket']
    else:
        top_labels = ['chair', 'table', 'office chair', 'cabinet', 'bookshelf', 'sofa', 'bed', 'monitor', 'storage cabinet', 'door']
    print("Top Labels:", top_labels)

    scenes = list_files(scene_data_filepath)


    for scene in scenes:

        scene_path = os.path.join(scene_data_filepath, scene)
        # Read .pth file
        scene_data = read_pth_file(scene_path)

        # Get the scene ID from the .pth file
        scene_id = scene_data['scene_id']
        # Get the segments_anno.json file for the scene
        segments_anno = get_segments_anno_file(scene_id)

        if segments_anno is not None:
            # Extract segGroups
            seg_groups = segments_anno.get('segGroups', [])
            array_path = os.path.join(mask_data_filepath, scene_id)
            mask_info_dict = {}

            print("scene", scene_id)
            if os.path.exists(array_path):
                for image_name in os.listdir(array_path):

                    image_filepath = os.path.join(array_path, image_name)
                    # Read .pkl file
                    array_data = read_pth_file(image_filepath)
                    
                    #Get instance ids that are visible in the current image
                    visible_instance_ids = np.unique(array_data)

                    # Filter segGroups based on labels to be used 
                    for label in top_labels:

                        #Get available instance ids from the scene based on a specific label
                        instance_ids = [group['id'] for group in seg_groups if group.get('label') == label]
                        #print("Instance IDs:", instance_ids)

                        # Create a new list with values present in both lists
                        intersection = [value for value in visible_instance_ids if value in instance_ids]

                        # Loop through the intersection list
                        for instance_id in intersection:
                            
                            pixel_count = count_pixels(instance_id, array_data)

                            key = f"{label}_{instance_id}"

                            if key not in mask_info_dict:
                                print("Key not in mask info dict", key)
                                mask_info_dict[key] = []

                            mask_info = {
                                'pixel_count': pixel_count,
                                'label': label,
                                'instance_id': instance_id,
                                'image_name': image_name
                            }
                            mask_info_dict[key].append(mask_info)
                for key, mask_info_list in mask_info_dict.items():
                    mask_info_list = sorted(mask_info_list, key=lambda x: x['pixel_count'], reverse=True)

                    if len(mask_info_list) >= 5:
                        top_5_masks = mask_info_list[:5]
                        print("Top 5 Masks:", top_5_masks)

                        for mask_info in top_5_masks:
                            mask_instance_id = mask_info['instance_id']
                            mask_label = mask_info['label']
                            mask_image_name = mask_info['image_name']
                            image_filepath = os.path.join(array_path, mask_image_name)
                            array_data = read_pth_file(image_filepath)
                            mask = create_mask(array_data, mask_instance_id)

                            new_filepath = os.path.join(mask_output_filepath, scene_id)
                            if not os.path.exists(new_filepath):
                                os.makedirs(new_filepath)

                            img_name_wo_ext = os.path.splitext(os.path.basename(mask_image_name))[0]
                            mask_label = mask_label.replace(" ", "-")
                            mask_name = f"{img_name_wo_ext}_{mask_label}_{mask_instance_id}.png"
                            mask_save_path = os.path.join(new_filepath, mask_name)

                            save_mask_as_png(mask, mask_save_path)