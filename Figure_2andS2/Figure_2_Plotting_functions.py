import glob as glob
import os
import pandas as pd
import numpy as np


def import_modelling_residuals(results):

    dfs_ = list()
    
    for r in results:
    
        files = list(glob.glob(os.path.join(r,"*")))
        files.sort(key=os.path.getmtime, reverse=True)
        d = pd.read_csv(glob.glob(os.path.join(files[0],"results/residuals.csv"))[0]).sort_values("residual")
        d['hyp'] = r.split("/")[3]
        dfs_.append(d)
    
    df = pd.concat(dfs_)

    df = df.loc[~df['hyp'].isin(["Free_1", "Free_2", "Free_3"])]

    return df

def select_best_residuals(df):

    df_best = list()
    
    for i in df.hyp.unique():
            z = df.loc[df['hyp']==i]
            z.sort_values('residual')
            df_best.append(z[0:1])
            
    df_best = pd.concat(df_best)

    return df_best

def calculate_aic(rss, num_params, num_observations):
    """
    Calculate AIC based on residual sum of squares (RSS), number of parameters, and number of observations.
    
    :param rss: Residual Sum of Squares (RSS) for the model
    :param num_params: Number of parameters in the model
    :param num_observations: Number of observations (residuals)
    :return: AIC value
    """
    return 2 * num_params + num_observations * np.log(rss / num_observations)

def plotting_modelling_heatmaps(top, hypo, data_path):

    sim = list(hypo.loc[hypo["Topology_number"]==top]['Topology_Name'])[0]
    d = pd.read_csv(glob.glob(os.path.join(data_path, 'results', sim,'*',"results/residuals.csv"))[0]).sort_values("residual")
    d['hyp'] = sim
    d = d.reset_index()
    
    path = glob.glob(os.path.join(data_path,'results', sim, '*',f'{sim}_fit_full'))
    file = open(path[0], 'rb')
    # dump information to that file
    data = pickle.load(file)
    
    custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', ['#FFFFFF', ingeo_colours[2], '#01544b'])
    data.plot_params["colour_dict"]["A"] = custom_cmap
    cmaps = [data.plot_params["colour_dict"][nm] for nm in data.data_names]
    k = d.loc[0]['index']
    a_unique, b_unique = np.unique(data.df["a"].values), np.unique(data.df["b"].values)
    extent, aspect = make_extent(a_unique, b_unique, "linear", "linear")
    vmax = data.plot_params["vmax"]
    
    
    k = d.loc[0]['index']
    cmaps = [data.plot_params["colour_dict"][nm] for nm in data.data_names]
    fig, ax = plt.subplots(2,len(data.data_names),sharex=True,sharey=True)
    for i, nm in enumerate(data.data_names):
        ax[1,i].imshow(np.flip(data.proportions_by_data_names[nm][k],axis=0),vmin=0,vmax=vmax,cmap=cmaps[i],aspect=aspect,extent=extent)
        ax[0,i].imshow(np.flip(data.true_final_vals_grid[nm],axis=0),vmin=0,vmax=vmax,cmap=cmaps[i],aspect=aspect,extent=extent)
        ax[0,i].set_title(nm)
    ax[0,0].set(ylabel="Data\nActivin")
    ax[1,0].set(ylabel="Sim\nActivin")
    for axx in ax[1]:
        axx.set(xlabel="BMP4")



##############################################
### For saving multiple heatmaps at once #####
##############################################

# import matplotlib.pyplot as plt
# import numpy as np

# # Assuming data is loaded and the necessary variables are defined already.
# a_unique, b_unique = np.unique(data.df["a"].values), np.unique(data.df["b"].values)
# extent, aspect = make_extent(a_unique, b_unique, "linear", "linear")
# vmax = data.plot_params["vmax"]

# k = d.loc[0]['index']
# cmaps = [data.plot_params["colour_dict"][nm] for nm in data.data_names]

# # Loop through each data name and save both top and bottom heatmaps
# for i, nm in enumerate(data.data_names):
#     # Create the top heatmap (first subplot)
#     fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figsize as needed
#     ax.imshow(np.flip(data.true_final_vals_grid[nm], axis=0), vmin=0, vmax=vmax, cmap=cmaps[i], aspect=aspect, extent=extent)
#     ax.set_xticks([])  # Remove x-axis ticks
#     ax.set_yticks([])  # Remove y-axis ticks
#     ax.set_xticklabels([])  # Remove x-axis labels
#     ax.set_yticklabels([])  # Remove y-axis labels

#     # Remove the title for top heatmap
#     # ax.set_title(f"Top Heatmap: {nm}", fontsize=10)  # No title

#     # Save the top heatmap as PNG with no border and no padding
#     plt.savefig(f"{sim}_{top}_{nm}_top_heatmap.png", dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)  # Close the top heatmap plot

#     # Create the bottom heatmap (second subplot)
#     fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figsize as needed
#     ax.imshow(np.flip(data.proportions_by_data_names[nm][k], axis=0), vmin=0, vmax=vmax, cmap=cmaps[i], aspect=aspect, extent=extent)
#     ax.set_xticks([])  # Remove x-axis ticks
#     ax.set_yticks([])  # Remove y-axis ticks
#     ax.set_xticklabels([])  # Remove x-axis labels
#     ax.set_yticklabels([])  # Remove y-axis labels

#     # Remove the title for bottom heatmap
#     # ax.set_title(f"Bottom Heatmap: {nm}", fontsize=10)  # No title

#     # Save the bottom heatmap as PNG with no border and no padding
#     plt.savefig(f"{sim}_{top}_{nm}_bottom_heatmap.png", dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)  # Close the bottom heatmap plot

