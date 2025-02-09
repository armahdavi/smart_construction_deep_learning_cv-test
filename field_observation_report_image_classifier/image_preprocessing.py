# -*- coding: utf-8 -*-
"""
Script to pre-process all training images prior to inputting to ANN models
It converts all images to 256*256 images by padding, and label thenm according to folder classifications 

@author: MahdaviAl
"""

from PIL import Image
import glob
import os
import numpy as np
from image_utils import list_image_formats, resize_and_pad_image, ten_crop
import h5py
import json
from sklearn.preprocessing import LabelEncoder


###############################################
### Step 1: Initial variables and functions ###
###############################################

target_size = 256 # image width and height size after padding
raw_folder = r'C:\python_projects\cnn\for_image_classifier\raw' # folder of all raw images (including folders of classes)
output_folder = r'C:\python_projects\cnn\for_image_classifier\processed\images' # folder of all processed images after padding
save_folder_array = r'C:\python_projects\cnn\for_image_classifier\processed' # folder of X and y arrays saved

def create_image_database(folder):
    '''
    Description
    ----------
    Function to create separate np arrays of training images and their labels.
    
    Parameters
    ----------
    folder : string folder from which input arrays X and output labels y are created.

    Returns
    -------
    X : np array of images for training
    y : np array of labels for trainng images.
    '''
    image_files = glob.glob(os.path.join(folder, '*.*'))
    image_files = [f for f in image_files if f.lower().endswith(('png', 'jpg', 'jpeg', 'webp', 'mpo'))]    
    
    X, y = [], []  # List to store the processed images and labels
        
    for image_file in image_files:
        # Load and preprocess each image
        image = Image.open(image_file)
        resized_image_array = np.array(image)
        
        file_name_lower = os.path.basename(image_file.lower())
                
        if 'drawing' in file_name_lower:
            label = 'drawing'
        elif 'logo' in file_name_lower:
            label = 'logo'
        elif 'field' in file_name_lower:
            label = 'field'
        elif 'signature' in file_name_lower:
            label = 'signature'
        else:
            label = 'unknown'
     
        
        # Add the processed image and label to the lists
        X.append(resized_image_array)
        y.append(label)
    
    # Convert X and y to NumPy arrays
    X = np.array(X)
    y = np.array(y)
        
    return X, y



def augment_dataset(images, labels, classes_to_augment, crop_ratio = 0.8):
    '''
    Description
    ----------
    Function to augment the dataset by applying ten-crop augmentation to specific classes 
    (good for imbalanced classes where data augmentation should be run over only those classes)
    
    Parameters
    ----------
    images : np array of image (RGB)
    labels : np array of labels (strings or integer)
    classes_to_augment : list of classes where augmentation is implemented
    crop_ratio : float number of the crop ratio (default = 0.8)
        
    Returns
    -------
    Augmented arrays of images and labels
    '''
    
    augmented_images, augmented_labels = [], [] # List to store the augmented images and labels
    
    for img, label in zip(images, labels):
        if label in classes_to_augment:
            # Apply ten-crop augmentation
            crops = ten_crop(img, crop_ratio)
            augmented_images.extend(crops)
            augmented_labels.extend([label] * len(crops))
        else:
            # Keep images of other classes unchanged
            augmented_images.append(img)
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)


############################################
### Step 2: Process raw images and save  ###
############################################

# Collect all image formats in a list
format_list = []
for fold_name in ['drawings', 'field', 'logo', 'signature']:
    test_folder = os.path.join(raw_folder, fold_name)
    format_list = list(set(format_list + list_image_formats(test_folder)))

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save processed images 
for item in ['drawings', 'field', 'logo', 'signature']:
    # Find all image files in the input directory
    png = glob.glob(os.path.join(raw_folder, item, r'*.png'))
    jpg = glob.glob(os.path.join(raw_folder, item, r'*.jpg'))
    jpeg = glob.glob(os.path.join(raw_folder, item, r'*.jpeg'))
    webp = glob.glob(os.path.join(raw_folder, item, r'*.webp'))
    mpo = glob.glob(os.path.join(raw_folder, item, r'*.mpo'))
    
    image_files = png + jpg + jpeg + webp + mpo
    ## I don't know why loop doesn't work above so I hard-coded
    
    i = 0
    for image_file in image_files:
        # Load and preprocess each image
        # image = Image.open(image_file)
        resized_image = resize_and_pad_image(image_file, target_size)  # Use the resize_and_pad_image function from the previous example
        
        # Save the preprocessed image to the output directory
        filename = os.path.basename(item + '_' + str(i))
        output_file = os.path.join(output_folder, filename + '.jpg')
        resized_image.save(output_file)
        i += 1
    
# Create database ready for further deep learning modeling
X, y = create_image_database(output_folder)

# Encode class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save labels and encoded as json (to call later if needed)
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
with open(os.path.join(save_folder_array, 'label_mapping.json'), 'w') as f:
    json.dump(label_mapping, f)

# Save the database
with h5py.File(os.path.join(save_folder_array, 'dataset.h5'), 'w') as hf:
    hf.create_dataset('X', data = X)
    hf.create_dataset('y', data = y)

