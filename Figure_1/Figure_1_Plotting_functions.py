## Plotting functions
import os
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib as mpl

import matplotlib.pyplot as plt
import seaborn as sns
from re import match

## Functions used in Inge et al. for plotting fate proportions and graphing.
ingeo_colours = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB", "#5D55A4"]

def sample_equally(dataframe, class_, sample_n):
    sample_list = list()
    for i in dataframe[class_].unique():
        df_ = dataframe.loc[dataframe[class_]==i]
        df_ = df_.sample(sample_n)
        sample_list.append(df_)
    df_ = pd.concat(sample_list)
    return df_

def remove_outliers(dataframe, col_name, class_):
    sample_list = list()
    for i in dataframe[class_].unique():
        df_in = dataframe.loc[dataframe[class_]==i]
        
        
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3-q1 #Interquartile range
        fence_low  = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]


        
        sample_list.append(df_out)
    df_ = pd.concat(sample_list)
    return df_

def dual_marker_thresholding(data_frame, mk1, marker1_name, mk1T, mk2, marker2_name, mk2T, conditions_to_split):
    
    En = len(conditions_to_split)-1

    # Combined cell type status
    subset = data_frame
    groups = "Condition"


    subset.loc[(subset[f'{mk1}'] >= mk1T) & (subset[f'{mk2}'] >= mk2T), 'cell_type_status'] = f'{marker1_name}+{marker2_name}+'  
    subset.loc[(subset[f'{mk1}'] <= mk1T) & (subset[f'{mk2}'] >= mk2T), 'cell_type_status'] = f'{marker1_name}-{marker2_name}+'  
    subset.loc[(subset[f'{mk1}'] >= mk1T) & (subset[f'{mk2}'] <= mk2T), 'cell_type_status'] = f'{marker1_name}+{marker2_name}-'  
    subset.loc[(subset[f'{mk1}'] <= mk1T) & (subset[f'{mk2}'] <= mk2T), 'cell_type_status'] = f'{marker1_name}-{marker2_name}-'  



    celltypes = subset.groupby(["Condition", 'cell_type_status']).count()
    celltypes = pd.DataFrame(celltypes['label']).reset_index().pivot(index=groups, columns='cell_type_status', values='label')
    cell_type_status = celltypes.fillna(0)

    counts = data_frame.groupby(["Condition"]).count()

    data = [counts["label"],
            cell_type_status[f'{marker1_name}+{marker2_name}-'],
            cell_type_status[f'{marker1_name}-{marker2_name}+'],
            cell_type_status[f'{marker1_name}+{marker2_name}+'],
            cell_type_status[f'{marker1_name}-{marker2_name}-']

           ]

    headers = ["Count", 
               f'{marker1_name}+{marker2_name}-',
               f'{marker1_name}-{marker2_name}+',
               f'{marker1_name}+{marker2_name}+',
               f'{marker1_name}-{marker2_name}-'
              ]

    data_pos = pd.concat(data, axis=1, keys=headers)
    data_pos = data_pos.reset_index()
    data_pos = data_pos.rename(columns={'index': 'Condition'})

    #data_pos[["A","B"]] = data_pos['Condition'].str.split('_', 1, expand=True)


    data_pos[f'{marker1_name}+{marker2_name}-'] = data_pos[f'{marker1_name}+{marker2_name}-']/data_pos["Count"]*100
    data_pos[f'{marker1_name}-{marker2_name}+'] = data_pos[f'{marker1_name}-{marker2_name}+']/data_pos["Count"]*100
    data_pos[f'{marker1_name}+{marker2_name}+'] = data_pos[f'{marker1_name}+{marker2_name}+']/data_pos["Count"]*100
    data_pos[f'{marker1_name}-{marker2_name}-'] = data_pos[f'{marker1_name}-{marker2_name}-']/data_pos["Count"]*100

   
    data_pos = data_pos.drop(columns=['Count'])
    
    return data_pos


def indivudal_marker(filtered_Sel, Marker_threshold, channel, marker_name):
    
    #Individual marker status
    groups = "Condition"
    subset = filtered_Sel

    subset.loc[(subset[channel] >= Marker_threshold) , 'Marker_status'] = 'Marker+'  
    subset.loc[(subset[channel] <= Marker_threshold) , 'Marker_status'] = 'Marker-'  

    celltypes = subset.groupby(["Condition", 'Marker_status']).count()
    celltypes = pd.DataFrame(celltypes['label']).reset_index().pivot(index=groups, columns='Marker_status', values='label')
    Marker_status = celltypes.fillna(0)

    counts = filtered_Sel.groupby(["Condition"]).count()

    data = [counts["label"],
            Marker_status["Marker+"],
            Marker_status["Marker-"],


           ]

    headers = ["Count", 
               "Marker+",
               "Marker-"
              ]

    data_pos = pd.concat(data, axis=1, keys=headers)
    data_pos = data_pos.reset_index()
    data_pos = data_pos.rename(columns={'index': 'Condition'})

    #data_pos[['Condition']] = data_pos['Condition'].str.split('_', 1, expand=True)


    data_pos["Marker+"] = data_pos["Marker+"]/data_pos["Count"]*100
    data_pos["Marker-"] = data_pos["Marker-"]/data_pos["Count"]*100

    data_pos['Condition'] = data_pos[['Condition']].agg('_'.join, axis=1)

    data_pos = data_pos[["Condition","Marker+"]]
    data_pos = data_pos.rename(columns={"Marker+": f"{marker_name}+"})


    return data_pos

def create_stack_bar_plot(
    df,
    df_error_bar=None,
    x_figSize=2.5,
    y_figSize=2.5,
    y_label=None,
    y_axis_start=0,
    y_axis_limit=None,
    color_pal=sns.color_palette(palette="Blues_r"),
    bar_width=0.8,
    x_label =None
):

    fig, ax = plt.subplots(figsize=(x_figSize, y_figSize))

    sns.set(style="ticks")

    ax = df.plot(
        kind="bar",
        stacked=True,
        color=color_pal,
        width=bar_width,
        ax=ax,
        yerr=df_error_bar,
        capsize=4,
    )
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    sns.despine(ax=ax)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    ax.tick_params(axis="both", which="major", pad=1)
    plt.setp(ax.spines.values(), linewidth=1)

    if not y_axis_limit == None:
        ax.set_ylim(top=y_axis_limit)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        reversed(handles), reversed(labels), bbox_to_anchor=(1, 1), loc="upper left"
    )

def stirrplot_figure(data_frame, rm_outliers, channel, xclass_, marker_name, colour, order_, fluor, threshold, threshold_v, ylim_set, ylimits):
    
    if rm_outliers == True:
        data_frame_s = remove_outliers(data_frame, channel, xclass_)
    else:
        data_frame_s = data_frame

    plt.figure(figsize=(4,4), dpi=100)

    sns.set_theme(style="ticks")



    # Plot the orbital period with horizontal boxes
    #sns.boxplot(x=xclass_, y=channel, data=data_frame_s,
    #           whis=[0, 100], width=.6, palette=colour, order=order).set(yscale="log")

    if threshold == True:
        plt.axhline(y=threshold_v, color='grey', linestyle='--')

    # Add in points to show each observation
    sns.stripplot(x=xclass_, y=channel, data=data_frame_s,
                  size=3,  palette=colour,  alpha=0.4, linewidth=0, order=order, rasterized=True).set(yscale="log")
    
    if ylim_set == True:
        plt.ylim(ylimits)
        
    plt.grid()  #just add this
    plt.title(f'Average {marker_name} expression')
    plt.xlabel(f'{xclass_}')
    plt.ylabel(f'log(mean intensity {marker_name}) (Alexa-{fluor})')
    plt.xticks(rotation=90)
    
    
    
def percentage_single_marker_figure(data_frame, xclass_, channel, marker_name, threshold_v, Marker_positive, colour, order):
    data_pos = indivudal_marker(data_frame, threshold_v, channel, marker_name)

    if Marker_positive == True:
        marker_status = "Marker+"
        stat = "+"
    else: 
        marker_status = "Marker-"
        stat = "-"



    # seaborn theme
    sns.set_theme(style="ticks")
    f, ax = plt.subplots(1, figsize=(4,4), dpi=100)
    sns.barplot(x=data_pos[xclass_], y=data_pos[f"{marker_name}+"], ax=ax, order=order, palette=colour)
    # remove espine
        # add labels
    container = ax.containers[0]
    labels = ["{0:.0f}%".format(val) for val in container.datavalues]
    ax.bar_label(container, labels=labels, padding=15)

    plt.ylim(0,100)
    plt.grid()  #just add this
    plt.title(f'{marker_name} %{stat}ve')
    plt.xlabel(f'{xclass_}')
    plt.ylabel(f'%{stat}ve')
    plt.xticks(rotation=90)
    
    

def Heatmap_figure(df, marker, colour_pal_sel, vmin_):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Filter to relevant columns
    df = df[['A', 'B', marker]]

    # Pivot to matrix form
    DF = df.pivot(index='A', columns='B', values=marker)

    # Correct sorting:
    DF = DF.sort_index(axis=0, key=lambda x: x.astype(int), ascending=False)  # A descending (y-axis)
    DF = DF.sort_index(axis=1, key=lambda x: x.astype(int), ascending=True)   # B ascending (x-axis)

    matrix = DF

    # Define color map
    boundaries = sorted(matrix.stack().tolist())
    colors = ["#FFFFFF", colour_pal_sel]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", colors)

    width = 0.3

    # Matplotlib styles
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.linewidth': width,
        'pdf.fonttype': 42
    })

    # Plot
    fig = plt.figure(figsize=(1.1, 1.1))
    ax = plt.subplot()

    g = sns.heatmap(matrix,
                    fmt=".0f",
                    ax=ax,
                    cmap=cmap,
                    cbar=False,
                    vmin=vmin_,
                    annot=True,
                    linewidths=width,
                    linecolor="Black")

    g.set_xlabel("[BMP4] in ng/mL")
    g.set_ylabel("[Activin] in ng/mL")
    plt.title(f'% {marker}')

    for _, spine in ax.spines.items():
        spine.set_visible(False)

    plt.tick_params(axis='both', which='major', labelsize=7, width=width, length=2)
    plt.tick_params(axis='both', which='minor', width=width * 0.666, length=2)

    plt.show()


# Continue with the heatmap function as you already have it

def Heatmap_figure_ratio(df, marker, colour_pal_sel, vmin_):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = df[['A', 'B', marker]]

    # Pivot to matrix form
    DF = df.pivot(index='A', columns='B', values=marker)

    # Fix row and column order:
    DF = DF.sort_index(axis=0, key=lambda x: x.astype(int), ascending=False)  # A descending (rows)
    DF = DF.sort_index(axis=1, key=lambda x: x.astype(int), ascending=True)   # B ascending (columns)

    matrix = DF.to_numpy()
    boundaries = sorted(matrix.flatten().tolist())

    colors = [ingeo_colours[0], "#FFFFFF", ingeo_colours[4]]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", colors)

    width = 0.3

    # Configure Matplotlib
    mpl.rcParams['font.family'] = 'Helvetica'
    mpl.rcParams['font.sans-serif'] = ['Helvetica']
    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.titlesize'] = 7
    mpl.rcParams['axes.labelsize'] = 7
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['legend.fontsize'] = 7
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['axes.labelcolor'] = 'black'
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.color'] = 'black'
    mpl.rcParams['axes.linewidth'] = width
    mpl.rcParams['pdf.fonttype'] = 'truetype'

    fig = plt.figure(figsize=(1.1, 1.1))
    ax = plt.subplot()

    annot_matrix = np.array([[f"{x:.2g}" for x in row] for row in matrix])

    g = sns.heatmap(matrix,
                    annot=annot_matrix,
                    fmt="",          # no default formatting, since annot is strings
                    ax=ax,
                    cmap=cmap,
                    cbar=False,
                    center=0,
                    cbar_kws={'format': '%002d', 'ticks': boundaries, 'drawedges': True},
                    xticklabels=True,
                    yticklabels=True,
                    linewidths=width,
                    linecolor="Black",
                    zorder=0)

    g.set_xticklabels(['0', '10', '25', '50', '100'])
    g.set_yticklabels(['100', '50', '25', '10', '0'])

    g.set_xlabel("[BMP4] in ng/mL")
    g.set_ylabel("[Activin] in ng/mL")

    plt.title(f'% {marker}')
    plt.tick_params(axis='both', which='major', labelsize=7, width=width, length=2)
    plt.tick_params(axis='both', which='minor', width=width * 0.666, length=2)

    plt.show()

    
    

def create_stack_bar_plot(
    df,
    df_error_bar=None,
    x_figSize=2.5,
    y_figSize=2.5,
    y_label=None,
    y_axis_start=0,
    y_axis_limit=None,
    color_pal=sns.color_palette(palette="Blues_r"),
    bar_width=0.8,
    x_label =None
):

    fig, ax = plt.subplots(figsize=(x_figSize, y_figSize))

    sns.set(style="ticks")

    ax = df.plot(
        kind="bar",
        stacked=True,
        color=color_pal,
        width=bar_width,
        ax=ax,
        yerr=df_error_bar,
        capsize=4,
    )
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    sns.despine(ax=ax)
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)
    ax.tick_params(axis="both", which="major", pad=1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.setp(ax.spines.values(), linewidth=1)

    if not y_axis_limit == None:
        ax.set_ylim(top=y_axis_limit)

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        reversed(handles), reversed(labels), bbox_to_anchor=(1, 1), loc="upper left"
    )