# -*- coding: utf-8 -*-
"""
Module to provides utility functions for efficient image preprocessing for field observation reports (FOR) image classification.

@author: MahdaviAl
"""

from PIL import Image
import os
import numpy as np
import pandas as pd
import tensorflow as tf


def list_image_formats(folder_path):
    '''
    Description
    -----------
    To list all image formats in a folder
    
    Parameters
    ----------
    folder_path : string of input folder
        
    Returns
    -------
    List of all image formats
    '''
    
    image_formats = set() # empty set to avoid format repeat
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # See if format is readable as image
        try:
            image = Image.open(file_path)
            image_format = image.format.lower()
            image_formats.add(image_format)
            image.close()
        except (IOError, SyntaxError):
            pass
    return list(image_formats)


def resize_and_pad_image(image_path, target_size):
    '''
    Description
    -----------
    Coonverts an input image to a specific square size by padding the missing area (e.g., 128*128, 256*256, etc.)
    
    Parameters
    ----------
    image_path : string of input image path
    target_size : size to which the image is converted
        
    Returns
    -------
    padded_image : converted image to the pads added
    '''
    
    image = Image.open(image_path)
    aspect_ratio = image.width / image.height
    
    # See if image is landscape or portrait
    if aspect_ratio > 1:
        # Landscape image
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        # Portrait or square image
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a new blank image, calculate position of resized image, and paste the image
    padded_image = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    padded_image.paste(resized_image, (left, top, right, bottom))
    
    return padded_image


def ten_crop(image, crop_ratio=0.8):
    '''
    Description
    -----------
    Generate 10 crops and re-size them to the original image size (for data augmentation)
    
    Parameters
    ----------
    image : np array of the image
    crop_ratio : ratio by which the image is cropped (default = 0.8).

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    height, width, _ = image.shape
    crop_h = int(height * crop_ratio)
    crop_w = int(width * crop_ratio)

    # Define crop coordinates
    crops = [
        image[:crop_h, :crop_w],                                # Top-left
        image[:crop_h, width - crop_w:],                        # Top-right
        image[(height - crop_h) // 2:(height + crop_h) // 2,    # Center
              (width - crop_w) // 2:(width + crop_w) // 2],
        image[height - crop_h:, :crop_w],                       # Bottom-left
        image[height - crop_h:, width - crop_w:],               # Bottom-right
    ]

    # Resize each crop to match the original size
    resized_crops = [tf.image.resize(crop, (height, width)) for crop in crops]

    # Generate flipped versions of each resized crop
    flipped_crops = [tf.image.flip_left_right(crop) for crop in resized_crops]

    return resized_crops + flipped_crops


def shuffle_data(X, y):
    '''
    Description
    -----------
    Shuffles trainin image set and their lables (of the same size)
    
    Parameters
    ----------
    X : np array of image set for training
    y : np array of image labels

    Returns
    -------
    X_shuffled : np array of image set for training after shuffle
    y_shuffled : np array of image labels
    '''
    
    assert len(X) == len(y), "X and y must have the same length."

    # Generate a random permutation of indices
    permutation = np.random.permutation(len(X))

    # Shuffle X and y arrays using the generated permutation
    X_shuffled = X[permutation]
    y_shuffled = y[permutation]

    return X_shuffled, y_shuffled

def get_normalized_counts(array, colname):
    '''
    Description
    -----------
    Create a dataframe with normalized abundance of each class in a label array
    
    Parameters
    ----------
    array : label array
    colname : the name assigned to the column after conversion to the df (with one column)

    Returns
    -------
    df : DataFrame of all class abundances plus all the examples
    '''
    unique, counts = np.unique(array, return_counts=True)
    normalized_counts = counts / counts.sum()
    df = pd.DataFrame({colname: normalized_counts}, index=unique)
    # Add a total row
    df.loc['total'] = counts.sum() 
    
    return df

