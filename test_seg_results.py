
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch 


def load_pickle_file(filepath):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def display_segmentation_image(segmentation_array):
    plt.imshow(segmentation_array, cmap='viridis')
    plt.title("Instance Segmentation Result")
    plt.axis('off')  # Hide the axis
    plt.savefig('/usr/prakt/s0090/scannetpp/presentation_plot.png')

# Path to the pickle file
filepath = "/storage/user/yez/scannet++/mask_data/8b5caf3398/DSC02580.pth"

# Load the segmentation result from the pickle file
# segmentation_array = load_pickle_file(filepath)
segmentation_array = torch.load(filepath)
#pth_data_dir = '/storage/user/yez/scannet++/pth_data/8b5caf3398.pth'
#pth_data = torch.load(pth_data_dir)
#obj_ids = np.unique(pth_data['vtx_instance_anno_id'])
#print(obj_ids)
# Check the type of the loaded data to ensure it's a numpy array
if isinstance(segmentation_array, np.ndarray):
    # Display the instance segmentation result image
    display_segmentation_image(segmentation_array)
    #print(segmentation_array)
else:
    print("The loaded data is not a numpy array. Please check the pickle file.")