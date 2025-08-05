#!/bin/bash

export LANG=en_GB.utf8
export LC_ALL=en_GB.utf8
unset LC_CTYPE

module purge
source ~/.bashrc   
conda activate stardist_py37
mkdir "overviews"


python - <<EOF

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os as os
import glob as glob
import napari

from skimage import data
from skimage.io import imread, imsave
#Path for data folder and segmented images
        
#Path for data folder and segmented images
        
data_path = os.path.join("data")
seg_path = os.path.join(os.getcwd(),"segmented_images")

############################################
##DEFINING FUNCTIONS 
###########################################

import itertools

def read_images_pwc(path, well, position, channel):

    read_img = imread(glob.glob(os.path.join(path, f"*--W*{well}--P*{position}--*{channel}*.tif"))[0])

    return read_img

def well_position_filter(dataframe, well_n, position_n):
    """Return a dataframe filtered by well number and position"""
    
    filtered_df_ = dataframe.loc[(dataframe['Well'] == int(well)) & (dataframe['Position'] == int(position)) ]

    return filtered_df_

def filter_seg_mask(mask_image, label_list):
    """Return a filtered image mask by a label list"""
    
    con_ = list(set(np.unique(mask_image.flatten().tolist())) - set(label_list))
    filtered_mask = np.where(np.in1d(mask_image.flatten(), con_),0 , mask_image.flatten())                         
    filtered_mask  = filtered_mask.reshape(mask_image.shape)
    
    return filtered_mask

def binarise_mask(mask_image, value):
    """Return a binarised image mask """
    
   
    bin_mask =  mask_image.flatten()
    bin_mask[bin_mask > 0] = value
    bin_mask  = bin_mask.reshape(mask_image.shape)
    
    
    return bin_mask



def read_images_pwc(path, well, position, channel):

    read_img = imread(glob.glob(os.path.join(path, f"*--W*{well}--P*{position}--*{channel}*.tif"))[0])

    return read_img

import cv2
import numpy as np

def downscaler(image, factor):
    res = cv2.resize(image, dsize=(342, 342), interpolation=cv2.INTER_CUBIC)
    return res

def Image_generator(files):
    """Return Image generator object"""

    for filename in files:
        img = imread(filename)
        yield img
        
        
def Image_generator_dsc(files):
    """Return Image generator object"""

    for filename in files:
        img = imread(filename)
        img = downscaler(img, 6)
        yield img
        
                         
def filename_matches(path, well, position, channel):
    """Return files that match channel features"""


    fns = sorted(glob.glob(os.path.join(path, f"*--W*{well}--P*{position}--*{channel}*.tif")))
    
    return fns

def dask_stacker(filenames):
    
    """return stack of dask arrays"""
    
    sample = imread(filenames[0])

    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in filenames]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]

    stack = da.stack(dask_arrays, axis=0)

    return stack 
    
def min_max_intensity(fns):
    pbar = ProgressBar()
    min_ = list()
    max_ = list()

    for i in pbar(fns):
        
        im = imread(f'{i}')
        min_.append(np.min(im))
        max_.append(np.max(im))
        del(im)    

    return (min(min_), max(max_))

def well_overview_array(list_, position_n):
    sq_ = np.sqrt(len(position_n))
    x = 0 
    y = int(sq_)
    
    alt = [list(range(0,y))[i] for i in range(len(list(range(0,y)))) if i % 2 != 0]

    cols_ = list()

    for i in range(0, int(sq_)):
        if i in alt:
            t_ = list_[x:y]
            cols_.append(np.concatenate(t_[::-1], axis=0))
        else:
            cols_.append(np.concatenate(list_[x:y], axis=0))

        x=x+int(sq_)
        y=y+int(sq_)
    over_v = np.concatenate((cols_), axis=1)
    
    return over_v

import warnings
warnings.filterwarnings('ignore')

## Spiral merger (types of spirals)...

def spiral_cw(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[0])        # take first row
        A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)

def spiral_ccw(A):
    A = np.array(A)
    out = []
    while(A.size):
        out.append(A[0][::-1])    # first row reversed
        A = A[1:][::-1].T         # cut off first row and rotate clockwise
    return np.concatenate(out)


def to_spiral(A):
    A = np.array(A)
    B = np.empty_like(A)
    B.flat[base_spiral(*A.shape)] = A.flat
    return B


#Change if clockwise or counter
def base_spiral(nrow, ncol):
    return spiral_cw(np.arange(nrow*ncol).reshape(nrow,ncol))[::-1]


##############################################
##CALCULATING POS LAYOUT
##############################################

feature_data = pd.read_csv("feature_data.csv")
wells = list(feature_data.Well.unique())



positional_list = list(range(7,12))
positional_list = positional_list + list(range(14,19))
positional_list = positional_list + list(range(21,26))
positional_list = positional_list + list(range(28,33))
positional_list = positional_list + list(range(35,40))
positional_list=["%02d" % x for x in positional_list]   

def well_overview_array(list_, position_n):
    sq_ = np.sqrt(len(position_n))
    x = 0 
    y = int(sq_)
    
    alt = [list(range(0,y))[i] for i in range(len(list(range(0,y)))) if i % 2 != 0]

    cols_ = list()

    for i in range(0, int(sq_)):
        if i in alt:
            t_ = list_[x:y]
            cols_.append(np.concatenate(t_[::-1], axis=0))
        else:
            cols_.append(np.concatenate(list_[x:y], axis=0))

        x=x+int(sq_)
        y=y+int(sq_)
    over_v = np.concatenate((cols_), axis=1)
    
    return over_v

import warnings

##############################################
###GENERATE AND SAVE MONTAGES...
##############################################

for i in wells:
    print(f"well number {i}")
    ##DAPI
    
    DAPI_fns = list()
    for z in positional_list:
        x = filename_matches(data_path, i, z, "DAPI")
        DAPI_fns.append(x)
    
    DAPI_fns = [item for sublist in DAPI_fns for item in sublist]
    naming = DAPI_fns[0].split('/').pop().split('--')

    DAPI_ov = well_overview_array(list(Image_generator_dsc(DAPI_fns)), DAPI_fns)

    imsave(f"overviews/{naming[0]}_{naming[1]}_{i}_DAPI_395.tiff", DAPI_ov)
    
   ##a647
    
    a647_fns = list()
    for z in positional_list:
        x = filename_matches(data_path, i, z, "647")
        a647_fns.append(x)
    
    a647_fns = [item for sublist in a647_fns for item in sublist]
    naming = a647_fns[0].split('/').pop().split('--')

    a647_ov = well_overview_array(list(Image_generator_dsc(a647_fns)), a647_fns)

    imsave(f"overviews/{naming[0]}_{naming[1]}_{i}_HAND1_647.tiff", a647_ov)

    ##a647
    
    GFP_fns = list()
    for z in positional_list:
        x = filename_matches(data_path, i, z, "GFP")
        GFP_fns.append(x)
    
    GFP_fns = [item for sublist in GFP_fns for item in sublist]
    naming = GFP_fns[0].split('/').pop().split('--')

    GFP_ov = well_overview_array(list(Image_generator_dsc(GFP_fns)), GFP_fns)

    imsave(f"overviews/{naming[0]}_{naming[1]}_{i}_GATA6_GFP.tiff", GFP_ov)
    

EOF
