"""
Plots all of the 2d Fm NLFs
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
plt.style.use("science")

import matplotlib as mpl

datDir = "./"
files = [datDir+f for f in os.listdir(datDir) if f.endswith("dat")]

#os.makedirs("../plots/2d_fm_localization/",exist_ok=True)
textboxProps = {"boxstyle":'round', "facecolor":'white', "alpha":0.8}

def plot_localization(df,ax,col):
    df2 = df.copy()
    df2["r"] = -df2["r"]
    
    df = pd.concat((df,df2),ignore_index=True,axis=0)
    df = df.sort_values(["r","z"],ignore_index=True)
    
    uniqueCoords = [np.unique(df["r"]), np.unique(df["z"])]
    pLoc = df[col].to_numpy().reshape([len(u) for u in uniqueCoords])
    
    # cmap = mpl.cm.jet.with_extremes(over='1')
    ax.contourf(*uniqueCoords,pLoc.T,levels=len(rng),
                cmap=cmap)
    
    # zi = np.ma.masked_less(pLoc.T, .1)
    # ax.contourf(*uniqueCoords,zi,levels=100,
    #             cmap='jet')
    #ax.contour(*uniqueCoords,zi,colors='black',levels=2)
    print(df[col].max())
    
    return None

spac = 0.05
rng = np.arange(spac,1,spac)
cmap = mpl.colors.ListedColormap(['white',]+[mpl.cm.jet(i) for i in rng])

data_files = sorted(glob.glob('localization_*.dat'))

fig, axArr = plt.subplots(nrows=2,ncols=3,sharex=True,sharey=True)

asymm1Df = pd.read_csv(data_files[0],sep="\s+")
asymm2Df = pd.read_csv(data_files[1],sep="\s+")
compact1Df = pd.read_csv(data_files[2],sep="\s+")
compact2Df = pd.read_csv(data_files[3],sep="\s+")
elongatedDf = pd.read_csv(data_files[4],sep="\s+")

plot_localization(asymm1Df,axArr[0,0],"localizarionN")
plot_localization(asymm2Df,axArr[0,1],"localizarionN")

plot_localization(elongatedDf,axArr[0,2],"localizarionN")

plot_localization(compact1Df,axArr[1,0],"localizarionN")
plot_localization(compact2Df,axArr[1,1],"localizarionN")


axArr[1,2].axis('off')

fig.supxlabel(r"$r$ (fm)")
fig.supylabel(r"$z$ (fm)",x=-.03)


axArr[0,0].text(0.10, 0.90, "(a)", transform=axArr[0,0].transAxes, 
                fontsize=10,
                verticalalignment='top',horizontalalignment="left",
                #bbox=textboxProps,
                zorder=100)
axArr[0,1].text(0.10, 0.90, "(b)", transform=axArr[0,1].transAxes, 
                fontsize=10,
                verticalalignment='top',horizontalalignment="left",
                #bbox=textboxProps,
                zorder=100)

axArr[0,2].text(0.10, 0.90, "(c)", transform=axArr[0,2].transAxes, 
                fontsize=10,
                verticalalignment='top',horizontalalignment="left",
                #bbox=textboxProps,
                zorder=100)

axArr[1,0].text(0.10, 0.90, "(d)", transform=axArr[1,0].transAxes, 
                fontsize=10,
                verticalalignment='top',horizontalalignment="left",
                #bbox=textboxProps
                zorder=100)

axArr[1,1].text(0.10, 0.90, "(e)", transform=axArr[1,1].transAxes, 
                fontsize=10,
                verticalalignment='top',horizontalalignment="left",
                #bbox=textboxProps,
                zorder=100)

lims = (-30,30)
for a in axArr.flatten():
    a.set_aspect(1)
    a.set(xlim=lims,ylim=lims)
    
fig.subplots_adjust(hspace=-0.16,wspace=0)

axArr[1,0].set(xticks=[-20,0,20])

fig.savefig("Z_122_N_184_NLFs.pdf")

# plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap,),
#               ax=axArr[1,2])

# for a in axArr.flatten():
#     a.axis('off')
# fig.savefig('colorbar.pdf')

