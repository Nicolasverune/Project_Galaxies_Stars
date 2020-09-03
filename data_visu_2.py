import pandas as pd
import numpy as np
from scipy.stats import  kurtosis, skew
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

galaxies = pd.read_csv("galaxies_clean.csv")
stars = pd.read_csv("stars_clean.csv")

#numpy mean

flux = ["u","g","r","i","z"]
color = ["u-g","g-r","r-i","i-z"]

def plot(DataPanda,hmm,df=None,stat=False):
    f, axes = plt.subplots(1, len(hmm), figsize=(7*len(hmm), 7), sharex=True)
    for i in range(len(hmm)) : 
        sns.distplot(DataPanda[hmm[i]], color="skyblue", ax=axes[i])
    
    
    if stat :
        for i, ax in enumerate(axes.reshape(-1)):
            sk = df.at[i,0]
            kr = df.at[i,1]
            mn = df.at[i,2]
            md = df.at[i,3]
            ax.text(x=0.97, y=0.97, transform=ax.transAxes, s="Skewness: %f" % sk,\
                fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
                backgroundcolor='white', color='xkcd:poo brown')
            ax.text(x=0.97, y=0.91, transform=ax.transAxes, s="Kurtosis: %f" % kr,\
                fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
                backgroundcolor='white', color='xkcd:dried blood')
            ax.text(x=0.97, y=0.85, transform=ax.transAxes, s="Median: %f" % mn,\
                fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
                backgroundcolor='white', color='xkcd:army green')
            ax.text(x=0.97, y=0.79, transform=ax.transAxes, s="Mean: %f" % md,\
                fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
                backgroundcolor='white', color='xkcd:eggplant')


    plt.tight_layout()
    

def create_diff(DataPanda,hmm) : 
    for i in range(len(hmm)-1) : 
        DataPanda[hmm[i]+"-"+hmm[i+1]] = DataPanda[hmm[i]] - DataPanda[hmm[i+1]]


def get_stat(hmm,*data):
    df = pd.DataFrame( columns = ['nm','sk','kr','mn','md','name'])
    ind = 0    
    for DataPanda, DataName in data :         
        for i in range(len(hmm)) :         
            sk = DataPanda[hmm[i]].skew()
            md = np.median(DataPanda[hmm[i]])
            kr = DataPanda[hmm[i]].kurt()
            mn = np.mean(DataPanda[hmm[i]])
            df.loc[ind] = [hmm[i],sk,kr,mn,md,DataName]
            ind += 1
    return df


def plot_stat(Data,title):
    fig, ax = plt.subplots(ncols=2, nrows=2)
    
    plt.title(title)    

    sns.pointplot(x = 'nm',y='mn', hue = Data['name'], data=Data, ax=ax[0][0])
    sns.pointplot(x = 'nm',y='kr', hue = Data['name'], data=Data, ax=ax[1][1])
    sns.pointplot(x = 'nm',y='sk', hue = Data['name'], data=Data, ax=ax[1][0])
    sns.pointplot(x = 'nm',y='md', hue = Data['name'], data=Data, ax=ax[0][1])
      
    ax[0][0].set_ylabel("mean")
    ax[0][1].set_ylabel("median")
    ax[1][0].set_ylabel("skew")
    ax[1][1].set_ylabel("kurtosis")

    ax[0][0].set_xlabel("")
    ax[0][1].set_xlabel("")
    ax[1][0].set_xlabel("")
    ax[1][1].set_xlabel("")

    plt.tight_layout()

def scatter_col_fl(data,color,flux):
    g = sns.PairGrid(data, hue="type",x_vars=flux,y_vars=flux)
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter,s=1)
    g.add_legend();
    g = sns.PairGrid(data, hue="type",x_vars=color,y_vars=color)
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter,s=1)
    g.add_legend();
    g = sns.PairGrid(data, hue="type",x_vars=color,y_vars=flux)
    g.map(plt.scatter,s=1)

    g.add_legend();

galaxies = pd.read_csv("galaxies_clean.csv")
stars = pd.read_csv("stars_clean.csv")
create_diff(galaxies,flux)
create_diff(stars,flux)
galaxies = galaxies.filter(['type','u','g','r','i','z',"u-g","g-r","r-i","i-z"],axis=1)
stars = stars.filter(['type','u','g','r','i','z',"u-g","g-r","r-i","i-z"],axis=1)

gal_star = galaxies.append(stars,ignore_index=True)

scatter_col_fl(gal_star,color,flux)

"""
create_diff(galaxies,flux)
create_diff(stars,flux)
        

flux_stat = get_stat(flux,(galaxies,'galaxies'),(stars,'stars'))
color_stat = get_stat(color,(galaxies,'galaxies'),(stars,'stars'))

print(flux_stat)

plot_stat(flux_stat,'flux statistics')
plot_stat(color_stat,'color statistics')


"""
plt.show()

