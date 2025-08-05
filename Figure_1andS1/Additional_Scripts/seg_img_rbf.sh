#!/bin/bash

export LANG=en_GB.utf8
export LC_ALL=en_GB.utf8
unset LC_CTYPE

module purge
source ~/.bashrc   
conda activate stardist_py37
mkdir "segmented_images"


python - <<EOF

from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams["image.interpolation"] = None
import os
from glob import glob

import matplotlib.pyplot as plt
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.utils import Path, normalize
from skimage.io import imread, imsave
from stardist import _draw_polygons, export_imagej_rois, random_label_cmap
from stardist.models import StarDist2D
from tifffile import imread
from skimage import measure, restoration
from skimage.segmentation import clear_border



np.random.seed(6)
lbl_cmap = random_label_cmap()

def flatten(t):
    return [item for sublist in t for item in sublist]

cwd = os.getcwd()

x = cwd.split("/")
folder = str(x.pop())


data_path_root = "/camp/lab/santoss/data/instruments/ScanR1/2022/Dec"
data_path = os.path.join(data_path_root, folder, "data")

output_path = cwd

channels = ['DAPI','GFP','mCherry','647']
seg_ch = ['DAPI']
rbf=25

def segmentation_generator(files):
    """Segmentation generator"""

    for filename in files:
        img = imread(filename)
        yield img

model = StarDist2D.from_pretrained("2D_versatile_fluo")

image_selection =  np.array([("*", ["*"])])

fns = []
imgs = []

print("initialised")


img_selection_list = []
for well in range(len(image_selection)):
    for components in range(1, len(image_selection[well])):
        for subcomp in range(0, len(image_selection[well][components])):
            img_selection_list.append(str(f"{image_selection[well][0]}--{image_selection[well][components][subcomp]}"))

for sel in img_selection_list:
    fns.append(sorted(glob(os.path.join(data_path, f"*{sel}*DAPI*.tif"))))

fns = flatten(fns)
imgs = segmentation_generator(fns)
imgs_l = list(imgs)

os.chdir(output_path)

print("loaded images")


for img, files in zip(imgs_l, fns):
    foo = normalize(img)
    predict_img, _ = model.predict_instances(foo)
    img_name = os.path.basename(files)
    print(img_name)
    imsave(os.path.join(output_path,f"segmented_images/{img_name}"),
        predict_img,
    )

print("segmented")


#Specify channels in folder to analyse.

#Parameters to measure
measured_pars = ['label', 'mean_intensity']

seg_path = os.path.join(output_path, "segmented_images")


##Form lazy load of segmented images
seg_fns = list(sorted(glob(os.path.join(seg_path,"*.tif"))))
seg_imgs = segmentation_generator(seg_fns)

seg_imgs = segmentation_generator(seg_fns)

channels.sort() 

img_data = pd.DataFrame( columns = ['Condition','Well','Position','Z','T','tif'])
parameter_data_list = []

for seg_img, seg_file in zip(seg_imgs, seg_fns):
    seg_oi = seg_file.replace(seg_path+"/","").replace("--DAPI.tif","")
   
    channels_fns = sorted(glob(os.path.join(data_path, str(seg_oi + "*.tif"))))
    channel_imgs = segmentation_generator(channels_fns)
    
    #object filtering...
    seg_img_p = clear_border(seg_img)
    
    #well_pos_data based on 
    measure_data_temp = []
    measure_data_temp.append(pd.DataFrame(measure.regionprops_table(seg_img_p, properties=['label', 'area','eccentricity','centroid'])))

    for ch_img, ch_file, ch_names in zip(channel_imgs, channels_fns, channels):
        ch_img = (ch_img - restoration.rolling_ball(ch_img, radius = rbf))
        measure_data_temp.append(pd.DataFrame(measure.regionprops_table(seg_img_p, ch_img, properties=measured_pars)).rename(columns={"mean_intensity": str("mean_intensity_" + ch_names)}))
    
    measure_data = measure_data_temp[0]
    
    for dataset_n in range(1, len(measure_data_temp)):
        measure_data = pd.merge(measure_data, measure_data_temp[dataset_n],on='label',how='left')
        
        
    parameter_data_list.append(measure_data)

    split = list(seg_file.split('--'))
    img_data_l = pd.DataFrame([split], columns = ['Condition','Well','Position','Z','T','tif'])
    Position = int(img_data_l.iloc[0]['Position'].replace("P000",""))
    Well = int(img_data_l.iloc[0]['Well'].replace("W000",""))
    
    measure_data['Position'] = Position
    measure_data['Well'] = Well
    measure_data['Condition'] = str(seg_oi).split("--W")[0]



feature_data = pd.concat(parameter_data_list, axis=0, join="outer")
feature_data.to_csv(os.path.join(output_path, "feature_data.csv"))

print("quantified - completed")

EOF
