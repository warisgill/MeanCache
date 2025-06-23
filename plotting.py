import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from diskcache import Index
import shlex

import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import pandas as pd
import seaborn as sns

import flwr as fl

# font_size = 14
# # mpl.use("pgf")
# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",  # or "xelatex" or "lualatex"
#     "font.family": "serif",
#     "text.usetex": True,
#     "pgf.rcfonts": False,
#     "axes.labelsize": font_size,      # Font size for x and y labels
#     "font.size": font_size,           # Global font size
#     "legend.fontsize": font_size,      # Font size for the legend
#     "xtick.labelsize": font_size,      # Font size for x-axis tick labels
#     "ytick.labelsize": font_size       # Font size for y-axis tick labels
# })





def barGraph(df, x, y, hue_col, hatches = ["||", "+"]):
    # Create bar plot
    def fmt(x):
        return f'{x:.2f}'

    fig =  plt.figure(figsize=(10, 6))
    ax_barplot = sns.barplot(x=x, y=y, hue=hue_col, data=df)
    ax_barplot.bar_label(ax_barplot.containers[0], fmt=fmt ) # first bar
    ax_barplot.bar_label(ax_barplot.containers[1], fmt= fmt) # second bar

    
    # Loop over the bars
    for bars, hatch in zip(ax_barplot.containers, hatches):
        # Set a different hatch for each group of bars
        for bar in bars:
            bar.set_hatch(hatch)
    
    # ax_barplot.legend(loc='upper center',  fancybox=True, shadow=True)
    ax_barplot.legend(loc='upper center',  fancybox=True, shadow=True)

    
    return plt


def comparisonWithGPTCacheHitMissRates():
    fname =  'csvs/plot_comparison_gptcache-tname-paraphrase-albert-small-v2-dname-dgptcache-clients_per_round-10-num_clients-20-batch_size-64-device-cuda-client_epochs-3-num_rounds-10-.csv'
    df = pd.read_csv(fname)    
    df = df[df['Metric'].isin(['True Hit Rate', 'False Hit Rate', 'True Miss Rate', 'False Miss Rate'])]
    df.dropna(inplace=True)
    plt =  barGraph(df, x="Metric", y="Value", hue_col="Cache Type")

    
    plt.tight_layout()
    plt.xlabel("Metric")
    plt.ylabel("Hit/Miss Rate")
    plt.savefig(f'graphs/pdfs/hitmiss_rate.png', bbox_inches='tight')

def comparisonWithGPTF1Scores():
    fname =  'csvs/plot_comparison_gptcache-tname-paraphrase-albert-small-v2-dname-dgptcache-clients_per_round-10-num_clients-20-batch_size-64-device-cuda-client_epochs-3-num_rounds-10-.csv'
    df = pd.read_csv(fname)    
    df = df[df['Metric'].isin(['F1', 'Precision', 'Recall'])]
    df.dropna(inplace=True)
    plt =  barGraph(df, x="Metric", y="Value", hue_col="Cache Type")

    
    plt.tight_layout()
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.savefig(f'graphs/pdfs/f1_precesion_recall.png', bbox_inches='tight')


def plotImpactofThreshold():
    def linePlot(axis, color_markers, title = None, cols=None):
        temp_ax = None
        for i, col in enumerate(cols):
            if col not in df.columns:
                continue
            color, marker = color_markers[i]
            
            temp_ax = sns.lineplot(x=df['Threshold'], y=df[col], color=color, marker=marker, label=col, legend='full', ax=axis)
            # else:
            #     temp_ax = sns.lineplot(x=df['Rounds'], y=df[col], color=color, marker=marker, ax=axis)
            
            # # Add text to the markers
            # i = 0
            # for x, y, text in zip(df['Rounds'], df[col], df[col]):
            #     if i%5 == 0:
            #         axis.annotate(round(text,2), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            #     i += 1
        
        if title is not None:
            temp_ax.set_title(title)
        
        temp_ax.set_ylabel("")
        return temp_ax 
    
    # all_color_markers = [('#1f77b4', 'v'), ('#2ca02c', '.'), ('#ff7f0e', 'x'), ('#9467bd', '+'), ('#8c564b', 's'), ('#e377c2', 'd'), ('#7f7f7f', '1'), ('#bcbd22', '2'), ('#17becf', '3')]
    colors = ["#E69F00", "#56B4E9", "#009E73", "#0072B2",]
    makers = ["_", "*", "o", "^"]
    
    
    fname = 'csvs/impactOfThreshold-tname-paraphrase-albert-small-v2-dname-dgptcache-clients_per_round-10-num_clients-20-batch_size-64-device-cuda-client_epochs-3-num_rounds-10-.csv'
    df = pd.read_csv(fname)
    cols = ['F1', 'Precision', 'Recall']
    print (cols)
    fig = plt.figure(figsize=(10, 6))
    linePlot(fig.gca(), color_markers= list(zip(colors, makers)), cols=cols)
    plt.savefig(f'graphs/pdfs/threshold1.png', bbox_inches='tight')
    plt.close()

    
    fig = plt.figure(figsize=(10, 6))
    cols = ['True Hit Rate', 'False Hit Rate', 'True Miss Rate', 'False Miss Rate']
    linePlot(fig.gca(), color_markers=list(zip(colors, makers)),  cols=cols)
    plt.savefig(f'graphs/pdfs/threshold2.png', bbox_inches='tight')
    plt.close()





    



if __name__ == '__main__':
    # comparisonWithGPTCacheHitMissRates() 
    # comparisonWithGPTF1Scores()
    plotImpactofThreshold()