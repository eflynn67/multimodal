import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rawDf = pd.read_csv("localization_asymm_Eshift_1.dat",sep="\s+")
reflectedDf = rawDf.copy()
reflectedDf["r"] = -1*reflectedDf["r"]

df = pd.merge(rawDf,reflectedDf,how="outer")
df = df.sort_values(["z","r"])

unVals = [np.unique(df["r"]),np.unique(df["z"])]
shp = [len(u) for u in unVals]

rr, zz = df["r"].to_numpy().reshape(shp), df["z"].to_numpy().reshape(shp)

for (i,col) in enumerate(df.columns[2:]):
    fig, ax = plt.subplots()
    cf = ax.contourf(rr,zz,df[col].to_numpy().reshape(shp),cmap="jet")
    ax.set(xlabel=r"$r$",ylabel=r"$z$",title="262Sg SKM* at OTP, "+col,
           xlim=(-10,10),ylim=(-20,20))
    
    plt.colorbar(cf,ax=ax)
    
    fig.savefig(col.replace("/","_over_")+".pdf",bbox_inches="tight")
    
df.to_csv("new_localization.dat",index=False,sep="\t")
