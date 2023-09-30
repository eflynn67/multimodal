import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
mode = 'asymm'
center_method = 'extent'
mass_dataFiles = sorted(glob.glob(f'mass_{mode}_Eshift_*_{center_method}.dat'))
charge_dataFiles = sorted(glob.glob(f'charge_{mode}_Eshift_*_{center_method}.dat'))

def read_fragments_out_file(inFile='fragments.out'):
    with open(inFile,'r') as fIn:
        lines = fIn.readlines()
    mass = []
    charge = []
    for l in lines:
        lSplit = l.replace('\n','').split()
        if 'mass' in lSplit:
            toAppend = mass
            continue
        if 'charge' in lSplit:
            toAppend = charge
            continue
        toAppend.append(np.array(lSplit).astype(float))
    mass = np.array(mass)
    charge = np.array(charge)
    return mass, charge

names = ['0.0','0.5','1.0']
massFig, massAx = plt.subplots()
chargeFig, chargeAx = plt.subplots()
color_arr = ['red','green','blue']
for i in np.arange(0,len(mass_dataFiles)):
    print(mass_dataFiles[i])
    print(charge_dataFiles[i])
    print(50*'=')
    mass = pd.read_csv(mass_dataFiles[i],sep='\t').to_numpy()
    charge = pd.read_csv(charge_dataFiles[i],sep='\t').to_numpy()
    #fragments.out has the mass/charge yields set up this way, for some reason
    #m1 = mass[:mass.shape[0]//2]
    #m2 = np.flip(mass[mass.shape[0]//2:],axis=0)
    
    massAx.plot(mass[:,0],mass[:,1],label=f'Eshift = {names[i]}',color=color_arr[i])
    massAx.plot(mass[:,0],mass[:,2],color=color_arr[i])
    massAx.fill_between(mass[:,0],mass[:,1],mass[:,2],hatch='/',facecolor='none',zorder=200,linewidth=0.0,\
                        edgecolor=color_arr[i])
    massAx.set(xlabel=r'$A$',title=f'Mass Yield (%) SkM* {center_method}')
    
    #c1 = charge[:charge.shape[0]//2]
    #c2 = np.flip(charge[charge.shape[0]//2:],axis=0)
    
    chargeAx.plot(charge[:,0],charge[:,1],label=f'Eshift = {names[i]}',color=color_arr[i])
    chargeAx.plot(charge[:,0],charge[:,2],color=color_arr[i])
    chargeAx.fill_between(charge[:,0],charge[:,1],charge[:,2],hatch='/',facecolor='none',zorder=200,linewidth=0.0,\
                        edgecolor=color_arr[i])
    chargeAx.set(xlabel=r'$A$',title=f'Charge Yield (%) SkM* {center_method}')
    
    #data = pd.read_csv(dataFile,nrows=169,sep='\s+',skiprows =1)
    #plt.plot(data.to_numpy()[:,0],data.to_numpy()[:,1],label=f'Eshift = {names[i]}')
massAx.legend()
chargeAx.legend()
massFig.savefig(f'256Fm_{mode}_mass_fragments_SkMstar_{center_method}.pdf')
chargeFig.savefig(f'256Fm_{mode}_charge_fragments_SkMstar_{center_method}.pdf')

plt.show()