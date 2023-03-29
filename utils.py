from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# this util module is for untility functions mainly used for ploting purposes
    
def plot_pics(pics, pics_name, n_cols=3, cell_size=4, figure_name=''):
    '''
    plot pictures in a chart

    Arguments
    ---------
    pics: (list) store the feature maps values - list elements are array (imgs)
    pics_name: (list) store the filter number of corresponding feature maps - list elements are int
    n_cols: (int) number of columns for subplot
    cell_size: (int) to construct cell size of subolot
    figure_name: (str) name og the fig
    '''  
    # create rows
    n_rows = ceil((len(pics)/n_cols))
    
    # create subplots
    fig, axes = plt.subplots(n_rows,n_cols, figsize=(cell_size*n_cols, cell_size*n_rows))

    # iterate over each subplot  
    for i,ax in enumerate(axes.flat):
        ax.grid(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # skip the subplot if there is no corresponding element to it
        if i<=len(pics_name)-1:
            ax.set_title(pics_name[i])
            ax.imshow(pics[i])

    # plot the model
    fig.suptitle(figure_name, fontsize='x-large', y=1.0)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)


def plot_mean_activations(nstrongest_filters, mean_filters_activations, figure_name=''):

    '''
    plot mean activations per filter and highlights the n most activated filters out
    
    Arguments
    ---------
    figure_name: (str) name og the fig
    nstrongest_filters: (int) indices of n stongest filters
    mean_filters_activations: (list) mean activation values of filters
    '''
    plt.figure()
    fig = plt.plot(np.clip(mean_filters_activations, 0., None), linewidth=1.8)
    axes = fig[0].axes

    for filter in nstrongest_filters:
        axes.axvline(x=filter, color='red', linestyle='--', alpha=0.4)

    axes.set_xlim(0, len(mean_filters_activations))

    axes.set_ylabel("Mean Activation")
    axes.set_xlabel("Filters")
    axes.set_title(figure_name)
    plt.show()