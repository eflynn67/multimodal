import sys, os
from scipy.interpolate import RBFInterpolator
import pandas as pd
import numpy as np
pyNebDir = os.path.expanduser("~/pyneb/src/pyneb/")
#pyNebDir = os.path.expanduser("~/ActionMinimization/py_neb")

if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
from utilities import *

nuc = '256Fm'
q30 = '24'
fName = f"./3d/slices/{nuc}/{nuc}_q30_{q30}.dat"
with open(fName) as fIn:
    headerLine = fIn.readline()
header = headerLine.lstrip("#").split()

df = pd.read_csv(fName,delimiter="\s+",names=header,skiprows=1)
df = df.rename(columns={"EHFB":"EHFB","M_22":"B2020","M_32":"B2030","M_33":"B3030"})

q20UnVals = np.unique(df["q20"])
q30UnVals = np.unique(df["q40"])
expectedMesh = np.meshgrid(q20UnVals,q30UnVals)
expectedFlat = np.array([[q2,q3] for q2 in q20UnVals for q3 in q30UnVals])

newDf = pd.DataFrame(data=expectedFlat,columns=["q20","q40"])

newDf = newDf.merge(df,on=["q20","q40"],how="outer")
newDf["is_interp"] = newDf["EHFB"].isna()

idxToInterp = newDf[newDf["is_interp"]==True].index
ptsToInterp = np.array(newDf[["q20","q40"]].iloc[idxToInterp])

interpCols = ["EHFB","B2020","B2030","B3030"]
for head in interpCols:    
    interp_func = RBFInterpolator(np.array(df[["q20","q40"]]),df[head])
    newDf[head].iloc[idxToInterp] = interp_func(ptsToInterp)

grids = np.meshgrid(q20UnVals,q30UnVals) 
zz = np.array(newDf["EHFB"]).reshape(expectedMesh[0].shape,order="F")

#minima_ind = SurfaceUtils.find_all_local_minimum(zz)
#gs_ind = SurfaceUtils.find_local_minimum(zz,searchPerc=[0.25,0.25],returnOnlySmallest=True)
#gs_coord = np.array((grids[0][gs_ind],grids[1][gs_ind])).T
#print(gs_coord)

#########
#E_gs = zz[gs_ind]
#print(E_gs)
#newDf["EHFB"] -= E_gs

h5File = h5py.File(f"./3d/slices/{nuc}/{nuc}_q30_{q30}.h5","w")
h5File.attrs.create("DFT","SKMs")
#h5File.attrs.create("GS_Inds",gs_ind)
#h5File.attrs.create("GS_Loc",[mesh[gs_ind] for mesh in expectedMesh])
#h5File.attrs.create("E_gs",E_gs)

h5File.create_group("interpolation")
h5File["interpolation"].attrs.create("method","scipy.interpolate.RBFInterpolator")
for col in newDf.columns:
    h5File.create_dataset(col,data=np.array(newDf[col]))
    

h5File.close()

