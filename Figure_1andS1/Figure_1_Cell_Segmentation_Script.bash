#!/bin/bash
#SBATCH --job-name=stardist_parallel
#SBATCH --output=logs/segmentation_%A_%a.out
#SBATCH --error=logs/segmentation_%A_%a.err
#SBATCH --array=0-1080            # Adjust the range to match the total number of images minus one
#SBATCH --cpus-per-task=1       # Adjust if you need more cores per task
#SBATCH --mem=16G               # Adjust memory as needed

export LANG=en_GB.utf8
export LC_ALL=en_GB.utf8
unset LC_CTYPE

module purge
source ~/.bashrc   
conda activate stardist_py37

# Usage:
#   For segmentation with StarDist:   sbatch cellpose_segment_and_quantify_parallel.sh segment
#   For quantification:                   sbatch cellpose_segment_and_quantify_parallel.sh quantify
#   For merging:                        sbatch cellpose_segment_and_quantify_parallel.sh merge
#
# SLURM_ARRAY_TASK_ID is automatically provided by SLURM.
mode=$1
if [ -z "$mode" ]; then
    echo "Usage: $0 <mode> where mode is 'segment', 'quantify', or 'merge'."
    exit 1
fi

python - "$mode" "$SLURM_ARRAY_TASK_ID" << 'EOF'
# --- Begin Python code ---
import sys
import os
import glob
import numpy as np
import pandas as pd
import skimage.io as io
from skimage import measure, restoration
from skimage.segmentation import clear_border

# --- StarDist imports ---
from csbdeep.utils import normalize
from stardist.models import StarDist2D

# ============
# Setup directories
# ============
cwd = os.path.dirname(os.path.abspath(__file__))

# Directory for segmented images.
seg_dir = os.path.join(cwd, "segmented_images")
os.makedirs(seg_dir, exist_ok=True)

# Directory for CSV outputs.
csv_output = os.path.join(cwd, "output")
os.makedirs(csv_output, exist_ok=True)

# Ensure logs directory exists.
logs_dir = os.path.join(cwd, "logs")
os.makedirs(logs_dir, exist_ok=True)

# ============
# Retrieve command-line arguments
# ============
if len(sys.argv) < 2:
    print("Usage: python <script> <mode> [task_index]")
    sys.exit(1)

mode = sys.argv[1]
if mode in ["segment", "quantify"]:
    try:
        task_index = int(sys.argv[2])
    except Exception as e:
        print("Error converting task index:", e)
        sys.exit(1)

# ============
# Common settings and functions
# ============
data_path = os.path.join(cwd, "data")
# List of known channels (for later quantification).
known_channels = ['DAPI', 'Alexa', 'GFP', 'mCherry']
rbf = 25

def get_channel_name(filename):
    """
    Extract the channel name from the filename.
    Expected format: <prefix>--<...>--<channel info>.tif,
    e.g., "Pluripotent--W00059--P00001--Z00000--T00000--Alexa 647.tif".
    """
    base = os.path.basename(filename)
    ch_part = base.split('--')[-1].replace('.tif','').strip()
    for known in known_channels:
        if known in ch_part:
            return known
    return ch_part

# ============
# Mode: segmentation using StarDist
# ============
if mode == "segment":
    dapi_files = sorted(glob.glob(os.path.join(data_path, "*DAPI*.tif")))
    if task_index >= len(dapi_files):
        print("Task index", task_index, "is out of range for segmentation tasks.")
        sys.exit(0)
    file = dapi_files[task_index]
    print("Segmenting file with StarDist:", file)
    
    # Read and normalize the image.
    try:
        img = io.imread(file)
        norm_img = normalize(img)
    except Exception as e:
        print("Error reading or normalizing file {}: {}".format(file, e))
        sys.exit(1)
    
    # Load the StarDist model.
    try:
        model = StarDist2D(None, '2D_versatile_fluo', basedir='/nemo/lab/santoss/home/users/ingeo/Analysis/2D_versatile_fluo')
    except Exception as e:
        print("Error loading StarDist model: {}".format(e))
        sys.exit(1)
    
    # Run segmentation.
    try:
        masks, _ = model.predict_instances(norm_img)
    except Exception as e:
        print("Error during StarDist prediction for {}: {}".format(file, e))
        sys.exit(1)
    
    # Save segmented image.
    img_name = os.path.basename(file)
    out_file = os.path.join(seg_dir, img_name)
    try:
        io.imsave(out_file, masks.astype(np.uint16))
    except Exception as e:
        print("Error saving segmented image to {}: {}".format(out_file, e))
        sys.exit(1)
    print("Segmentation completed for:", img_name)
    
# ============
# Mode: quantification
# ============
elif mode == "quantify":
    seg_files = sorted(glob.glob(os.path.join(seg_dir, "*.tif")))
    if task_index >= len(seg_files):
        print("Task index", task_index, "is out of range for quantification tasks.")
        sys.exit(0)
    seg_file = seg_files[task_index]
    print("Quantifying file:", seg_file)
    seg_img = io.imread(seg_file)
    seg_oi = os.path.basename(seg_file).replace("--DAPI.tif", "")
    channels_fns = sorted(glob.glob(os.path.join(data_path, f"{seg_oi}*.tif")))
    seg_img_p = clear_border(seg_img)
    measured_pars = ['label', 'mean_intensity']
    measure_data_temp = [
        pd.DataFrame(measure.regionprops_table(seg_img_p,
                                                properties=['label', 'area', 'eccentricity', 'centroid']))
    ]
    for ch_file in channels_fns:
        ch_img = io.imread(ch_file)
        ch_img_corrected = ch_img - restoration.rolling_ball(ch_img, radius=rbf)
        ch_name = get_channel_name(ch_file)
        temp_df = pd.DataFrame(
            measure.regionprops_table(seg_img_p, ch_img_corrected, properties=measured_pars)
        ).rename(columns={"mean_intensity": f"mean_intensity_{ch_name}"})
        measure_data_temp.append(temp_df)
    measure_data = measure_data_temp[0]
    for df in measure_data_temp[1:]:
        measure_data = pd.merge(measure_data, df, on='label', how='left')
    split = os.path.basename(seg_file).split('--')
    if len(split) >= 6:
        Condition, Well, Position, Z, T, _ = split[:6]
    else:
        Condition, Well, Position, Z, T = ("NA", "NA", "NA", "NA", "NA")
    Position_val = int(Position.replace("P000", "")) if Position != "NA" else -1
    Well_val = int(Well.replace("W000", "")) if Well != "NA" else -1
    measure_data['Position'] = Position_val
    measure_data['Well'] = Well_val
    measure_data['Condition'] = seg_oi.split("--W")[0]
    out_csv = os.path.join(csv_output, f"feature_data_{task_index}.csv")
    measure_data.to_csv(out_csv, index=False)
    print("Quantification completed for:", seg_file)
    
# ============
# Mode: merge
# ============
elif mode == "merge":
    merged_files = sorted(glob.glob(os.path.join(csv_output, "feature_data_*.csv")))
    if not merged_files:
        print("No 'feature_data_*.csv' files found to merge in", csv_output)
        sys.exit(0)
    df_list = [pd.read_csv(f) for f in merged_files]
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    out_file = os.path.join(csv_output, "feature_data.csv")
    merged_df.to_csv(out_file, index=False)
    print("Merged", len(merged_files), "files into", out_file)
    
    # --- Delete individual CSV files ---
    for f in merged_files:
        try:
            os.remove(f)
            print("Deleted intermediate CSV:", f)
        except Exception as e:
            print("Error deleting file", f, ":", e)
    
else:
    print("Unknown mode. Please use 'segment', 'quantify', or 'merge'.")
    sys.exit(1)
# --- End Python code ---
EOF
