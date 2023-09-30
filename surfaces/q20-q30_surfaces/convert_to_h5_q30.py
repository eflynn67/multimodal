import sys, os
from scipy.interpolate import RBFInterpolator
import pandas as pd
import numpy as np
pyNebDir = os.path.expanduser("~/pyneb/src/pyneb/")

if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
from utilities import *
edf='skmstar'
nuc = '254Fm'
fName = f"./{edf}/{nuc}.dat"
with open(fName) as fIn:
    headerLine = fIn.readline()
header = headerLine.lstrip("#").split()
print(header)
df = pd.read_csv(fName,delimiter="\s+",names=header,skiprows=1)

print(df)
#df = df.drop(labels=['E0','Z','N'],axis=1)
#print(df)
#df = df.rename(columns={"BE":"EHFB","M22":"B2020","M32":"B2030","M33":"B3030","Q20":'q20',"Q30":'q30'})
df = df.rename(columns={"EHFB":"EHFB","M_22":"B2020","M_32":"B2030","M_33":"B3030","Q20":'q20',"Q30":'q30'})
#df = df.rename(columns={"HFBROT":"EHFB","B_Q2":"B2020","B_Q2Q3":"B2030","B_Q3":"B3030","Q20":'q20',"Q30":"q30"})


# conversions for sam surfaces
#df['q20'] = df['q20']*2/(10**2)
#df['q30'] = df['q30']/10**3
#df['B2020'] =  df['B2020']*10**4 
#df['B2030'] =  df['B2030']*10**5 
#df['B3030'] =  df['B3030']*10**6 

q20UnVals = np.unique(df["q20"])
q30UnVals = np.unique(df["q30"])
expectedMesh = np.meshgrid(q20UnVals,q30UnVals)
expectedFlat = np.array([[q2,q3] for q2 in q20UnVals for q3 in q30UnVals])

df = df.dropna(axis=0)
newDf = pd.DataFrame(data=expectedFlat,columns=["q20","q30"])

newDf = newDf.merge(df,on=["q20","q30"],how="outer")

newDf["is_interp"] = newDf["EHFB"].isna()

idxToInterp = newDf[newDf["is_interp"]==True].index
ptsToInterp = np.array(newDf[["q20","q30"]].iloc[idxToInterp])

interpCols = ["EHFB","B2020","B2030","B3030"]
for head in interpCols:    
    interp_func = RBFInterpolator(np.array(df[["q20","q30"]]),df[head])
    newDf[head].iloc[idxToInterp] = interp_func(ptsToInterp)

grids = np.meshgrid(q20UnVals,q30UnVals) 
zz = np.array(newDf["EHFB"]).reshape(expectedMesh[0].shape,order="F")

minima_ind = SurfaceUtils.find_all_local_minimum(zz)
gs_ind = SurfaceUtils.find_local_minimum(zz,searchPerc=[0.7,0.25],returnOnlySmallest=True)
gs_coord = np.array((grids[0][gs_ind],grids[1][gs_ind])).T
print(gs_coord)

#########
#E_gs = zz[gs_ind]
#print(E_gs)
#newDf["EHFB"] -= E_gs

h5File = h5py.File(f"./{edf}/{nuc}.h5","w")
h5File.attrs.create("DFT","D1S")
#h5File.attrs.create("GS_Inds",gs_ind)
#h5File.attrs.create("GS_Loc",[mesh[gs_ind] for mesh in expectedMesh])
#h5File.attrs.create("E_gs",E_gs)

h5File.create_group("interpolation")
h5File["interpolation"].attrs.create("method","scipy.interpolate.RBFInterpolator")
for col in newDf.columns:
    h5File.create_dataset(col,data=np.array(newDf[col]))

h5File.close()

