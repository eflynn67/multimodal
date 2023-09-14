import sys, os
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import pandas as pd
import numpy as np
pyNebDir = os.path.expanduser("~/pyneb/src/pyneb/")
#pyNebDir = os.path.expanduser("~/ActionMinimization/py_neb")

if pyNebDir not in sys.path:
    sys.path.insert(0,pyNebDir)
from utilities import *
pd.set_option('display.max_columns', 7)
nuc = '258Fm'
fName = f'3d/{nuc}_UNEDF1.dat' #f"3d/{nuc}_SkMstar.dat"
with open(fName) as fIn:
    headerLine = fIn.readline()
header = headerLine.lstrip("#").split()

df = pd.read_csv(fName,delimiter="\t",names=header,skiprows=1)
df = df.rename(columns={"EHFB":"EHFB","M_22":"B2020","M_32":"B2030","M_42":"B2040","M_33":"B3030","M_43":"B3040","M_44":"B4040"})

q20UnVals = np.unique(df["Q20"])
q30UnVals = np.unique(df["Q30"])
q40UnVals = np.unique(df["Q40"])
#print('q20vals ',q20UnVals,len(q20UnVals))
#print('q30vals ',q30UnVals,len(q30UnVals))
#print('q40vals ',q40UnVals,len(q40UnVals))

expectedMesh = np.meshgrid(q20UnVals,q30UnVals,q40UnVals)
expectedFlat = np.array([[q2,q3,q4] for q2 in q20UnVals for q3 in q30UnVals for q4 in q40UnVals])




newDf = df.copy()


newDf["is_interp"] = newDf["EHFB"].isna()

df["is_interp"] = df["EHFB"].isna()

idxToInterp = df[df["is_interp"]==True].index
ptsToInterp = np.array(df[["Q20","Q30","Q40"]].iloc[idxToInterp])

newDf = newDf.dropna(axis=0,how='any')
print(newDf)
interpCols = ["EHFB","B2020","B2030","B2040","B3030","B3040","B4040"]


for head in interpCols:
    interp_func = RBFInterpolator(np.array(newDf[["Q20","Q30","Q40"]]),newDf[head],neighbors=200)
    interp_val = interp_func(ptsToInterp)
    df[head].iloc[idxToInterp] = interp_val
    

h5File = h5py.File(f"./3d/{nuc}_UNEDF1.h5","w")
h5File.attrs.create("DFT","SKMs")

h5File.attrs.create("interp_method","scipy.interpolate.RBFInterpolator")
for col in df.columns:
    h5File.create_dataset(col,data=np.array(df[col]))

h5File.close()
