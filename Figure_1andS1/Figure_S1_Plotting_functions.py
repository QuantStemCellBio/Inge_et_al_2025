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




def reorder_for_lp(df_, order_by, order_):
    
    df_['order'] = 0
    count = 0 
    for i in order_:
        df_.loc[(df_[order_by] == i), "order"] = count
        count = count + 1
    
    return df_.sort_values(['order'])

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

    data_pos[conditions_to_split] = data_pos['Condition'].str.split('_', En, expand=True)


    data_pos[f'{marker1_name}+{marker2_name}-'] = data_pos[f'{marker1_name}+{marker2_name}-']/data_pos["Count"]*100
    data_pos[f'{marker1_name}-{marker2_name}+'] = data_pos[f'{marker1_name}-{marker2_name}+']/data_pos["Count"]*100
    data_pos[f'{marker1_name}+{marker2_name}+'] = data_pos[f'{marker1_name}+{marker2_name}+']/data_pos["Count"]*100
    data_pos[f'{marker1_name}-{marker2_name}-'] = data_pos[f'{marker1_name}-{marker2_name}-']/data_pos["Count"]*100

   
    data_pos = data_pos.drop(columns=['Count'])
    
    return data_pos, subset



def processing_gated_plot(df):

    df[['B', 'A']] = df['Condition'].str.split(' ', 2, expand=True)
    df.B = df.B.map(lambda x: x.lstrip('B').rstrip('aAbBcC'))
    df.A = df.A.map(lambda x: x.lstrip('A').rstrip('aAbBcC'))
    df.A = df.A.astype(int)
    df.B = df.B.astype(int)

    conditions = ['B50 A0']

    df_ = list()
    for i in conditions:
        dfz = df.loc[df['Condition']==i]
        dfz = dfz.sample(n=1000, random_state=1)
        df_.append(dfz)
    df_ = pd.concat(df_)

    return df_, df, conditions


def contour_gates(df_, x, y, hue, palette, hue_order, dual_all):

    plt.figure(figsize=(4,4), dpi=100)
    sns.set_theme(style="ticks")


    sns.scatterplot(data=df_, x=x, y=y, alpha=0.3, hue=hue, palette=palette, hue_order=hue_order).set(xscale="log",yscale="log", xlim=(1.1,1000), ylim=(0.5,1000))
    sns.kdeplot(bw_adjust=1,thresh=0.05,levels=7, data=df_, x=x, y=y, color="grey").set(xscale="log",yscale="log", xlim=(1.1,1000), ylim=(0.5,1000))
    plt.axvline(x=16, color='grey', linestyle='--', alpha=0.5)
    plt.axhline(y=3.5, color='grey', linestyle='--', alpha=0.5)

    plt.xlabel(f'log(mean intensity HAND1) (Alexa-488)')
    plt.ylabel(f'log(mean intensity GATA6) (Alexa-647)')

    plt.text(1.4, 640, f'{dual_all[1]}%', fontsize=12)
    plt.text(1.4, 0.8, f'{dual_all[3]}%', fontsize=12)

    plt.text(250, 640, f'{dual_all[2]}%', fontsize=12)
    plt.text(250, 0.8, f'{dual_all[0]}%', fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()


def Heatmap_figure(df, marker, colour_pal_sel, vmin_):
    import matplotlib as mpl
    df = df[['A','B',marker]]
    DF = df.pivot(index='A', columns='B', values=marker)
    DF.sort_index(level=0, ascending=False, inplace=True)
    matrix = DF.to_numpy()
    boundaries = [value for value in matrix.flatten().tolist()]
    list.sort(boundaries)
    colors = [ "#FFFFFF", colour_pal_sel]
    
    #norm = matplotlib.colors.BoundaryNorm(boundaries=boundaries + [boundaries[-1]], ncolors=256)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

    
    width = 0.3
    
    # Configure Matplotlib to use Helvetica
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['font.size'] = 7
    
    # Set other font properties to maintain consistency
    mpl.rcParams['axes.titlesize'] = 7
    mpl.rcParams['axes.labelsize'] = 7
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7
    mpl.rcParams['legend.fontsize'] = 7
    
    
    # Explicitly set color to ensure correct rendering
    mpl.rcParams['axes.edgecolor'] = 'black'  # Ensure axis lines are black
    mpl.rcParams['axes.labelcolor'] = 'black' # Ensure axis labels are black
    mpl.rcParams['xtick.color'] = 'black'     # Ensure x-axis ticks are black
    mpl.rcParams['ytick.color'] = 'black'     # Ensure y-axis ticks are black
    
    mpl.rcParams['axes.linewidth'] = width  # Global setting
    
    # Ensure text is saved as text in PDF
    mpl.rcParams['pdf.fonttype'] = 'truetype'  # For PDF

    fig = plt.figure(figsize=(1.1, 1.1))
    ax = plt.subplot()

    


    g = sns.heatmap(matrix,
                fmt=".0f",
                ax=ax,
                cmap=cmap,
                #norm=norm,
                cbar=False, vmin=vmin_,
                cbar_kws={'format': '%02d', 'ticks': boundaries, 'drawedges': True},
                xticklabels=True,
                yticklabels=True, annot=True, linewidths=width, linecolor="Black")

    g.set_xticklabels(['0','10','25','50','100'])
    g.set_yticklabels(['100','50','25','10','0'])

    g.set_xlabel("[BMP4] in ng/mL")
    g.set_ylabel("[Activin] in ng/mL")

    plt.title(f'% {marker}')

     # Ensure grid lines are on top
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    plt.tick_params(axis='both', which='major', labelsize=7, width=width, length=2)  # Set major tick width
    plt.tick_params(axis='both', which='minor', width=width*0.666, length=2)  # Set minor tick width if you use minor ticks








   # plt.savefig(f'For_Paper_Aug_Replot_Perc_{marker}.pdf', dpi=72)
    plt.show()
    
    
def plotting_max_dose(df): 

    df_long = df.melt(
        id_vars=["Signal_cond"],
        value_vars=["HAND1+GATA6-", "HAND1-GATA6+", "HAND1+GATA6+", "HAND1-GATA6-","GATA6+"],
        var_name="Marker_Combo",
        value_name="Percentage"
    )


    df_long = df_long.loc[df_long['Signal_cond'].isin(["A100_B0",
                                             "A100_B10",
                                             "A100_B25",
                                             "A100_B50",
                                             "A100_B100",
                                             "A50_B100",
                                             "A25_B100",
                                             "A10_B100",
                                             "A0_B100"])]


    # Define your desired Signal_cond order
    signal_order = [
        "A100_B0", "A100_B10", "A100_B25", "A100_B50", "A100_B100",
        "A50_B100", "A25_B100", "A10_B100", "A0_B100"
    ]

    # Set both columns as ordered categoricals
    df_long["Signal_cond"] = pd.Categorical(df_long["Signal_cond"], categories=signal_order, ordered=True)

    # Already defined earlier
    marker_order = ["HAND1+GATA6-", "HAND1-GATA6+", "HAND1+GATA6+", "HAND1-GATA6-","GATA6+"]
    df_long["Marker_Combo"] = pd.Categorical(df_long["Marker_Combo"], categories=marker_order, ordered=True)

    # Sort by both
    df_sorted = df_long.sort_values(by=["Signal_cond", "Marker_Combo"]).reset_index(drop=True)

    df_long["Percentage_scaled"] = df_long.groupby("Marker_Combo")["Percentage"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())

    )

    pal = [ingeo_colours[4], ingeo_colours[0], ingeo_colours[2], ingeo_colours[7], ingeo_colours[6]]
    hue_order = ["HAND1+GATA6+","HAND1-GATA6+","HAND1+GATA6-","GATA6+","HAND1-GATA6-"]

    plt.figure(figsize=(5,4), dpi=80)
    sns.set_theme(style="ticks")

    sns.lineplot(data=df_long, x="Signal_cond", y="Percentage_scaled", hue="Marker_Combo", hue_order=hue_order, palette=pal)

    plt.title( "Percentage Cell Types", fontsize=15, weight='bold')

    plt.xlabel(f'Condition')


    plt.ylabel(f'normalised (% celltype)')
    plt.xticks(rotation=90)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    

def process_addswitch(df, SwDir_, Marker):
    if SwDir_ == 'AtB':
        Subset = df[df['Marker'] == Marker]
        Subset = Subset[Subset['Treatment'].isin(['ActivintoBMP4', 'BMP4', 'Activin', 'mTeSR'])]
        order = ['mTeSR_0hr', 'BMP4_S', 'ActivintoBMP4_6hrs', 'ActivintoBMP4_24hrs', 'ActivintoBMP4_48hrs', 'Activin_S']
    
    elif SwDir_ == 'BtA':
        Subset = df[df['Marker'] == Marker]
        Subset = Subset[Subset['Treatment'].isin(['BMP4toActivin', 'BMP4', 'Activin', 'mTeSR'])]
        order = ['mTeSR_0hr', 'Activin_S', 'BMP4toActivin_6hrs', 'BMP4toActivin_24hrs', 'BMP4toActivin_48hrs', 'BMP4_S']
        
    return Subset, order

def barplot_fs(df, ylim, order, hue, Marker):

    sns.set_theme(style="ticks")
    f, ax = plt.subplots(1, figsize=(4,4), dpi=50)

    sns.barplot(data=df, x='Condition', y='Marker+', order=order, capsize=0.4, color=hue)

    container = ax.containers[0]
    plt.title(f'{Marker}+')
    plt.ylabel(f'%Positive')
    plt.xticks(rotation=90)
    plt.ylim(0,ylim)