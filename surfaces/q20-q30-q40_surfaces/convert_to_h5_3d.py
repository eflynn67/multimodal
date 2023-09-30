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
edf ='skms'
fName = f'{edf}/{nuc}_raw.dat'
with open(fName) as fIn:
    headerLine = fIn.readline()
header = headerLine.lstrip("#").split()

df = pd.read_csv(fName,delimiter="\s+",names=header,skiprows=1)
df = df.rename(columns={"EHFB":"EHFB","M_22":"M_22","M_32":"M_23","M_42":"M_24","M_33":"M_33","M_43":"M_34","M_44":"M_44"})

q20UnVals = np.unique(df["expected_q20"])
q30UnVals = np.unique(df["expected_q30"])
q40UnVals = np.unique(df["expected_q40"])
df = df.dropna(axis=0)
expectedMesh = np.meshgrid(q20UnVals,q30UnVals,q40UnVals)
expectedFlat = np.array([[q2,q3,q4] for q2 in q20UnVals for q3 in q30UnVals for q4 in q40UnVals])

newDf = pd.DataFrame(data=expectedFlat,columns=["expected_q20","expected_q30","expected_q40"])


newDf = newDf.merge(df,on=["expected_q20","expected_q30","expected_q40"],how="outer")


#newDf = newDf.rename(columns={"expected_q20":'q20',"expected_q30":'q30',"expected_q40":'q40'})
newDf["is_interp"] = newDf["EHFB"].isna()

idxToInterp = newDf[newDf["is_interp"]==True].index

ptsToInterp = np.array(newDf[["expected_q20","expected_q30","expected_q40"]].iloc[idxToInterp])

interpCols = ["EHFB","M_22","M_23","M_24","M_33","M_34","M_44"]

for head in interpCols:
    interp_func = RBFInterpolator(df[["expected_q20","expected_q30","expected_q40"]].to_numpy(),df[head],neighbors=500)
    newDf[head].iloc[idxToInterp] = interp_func(ptsToInterp)

newDf.to_csv(f'{edf}/{nuc}.dat',sep=',')    
'''
h5File = h5py.File(f"./{nuc}.dat","w")
h5File.attrs.create("DFT","SKMs")

h5File.attrs.create("interp_method","scipy.interpolate.RBFInterpolator")
for col in df.columns:
    h5File.create_dataset(col,data=np.array(df[col]))

h5File.close()
'''