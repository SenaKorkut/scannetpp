
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
    plt.savefig('/usr/prakt/s0090/scannetpp/example_plot_3.png')

# Path to the pickle file
filepath = "/usr/prakt/s0090/segmentation_scene_results/DSC00633_inst.pkl"

# Load the segmentation result from the pickle file
segmentation_array = load_pickle_file(filepath)
pth_data_dir = '/usr/prakt/s0090/demo_annotations/98b4ec142f.pth'
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