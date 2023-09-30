import numpy as np
import matplotlib.pyplot as plt

from datetime import date
import sys
import os
from scipy import interpolate
import glob
### add pyneb
pyneb_path = os.path.expanduser('~/pyneb/src/pyneb') 
utils_path = os.path.expanduser('~/multimodal/utilities')
sys.path.insert(0, pyneb_path)
sys.path.insert(0, utils_path)
import utilities
import pandas as pd


#plt.style.use('science')
fdTol = 10**(-8)

if __name__ == "__main__":
    
    today = date.today()
    ### Define nucleus data path (assumes our github structure)
    edf = 'skms'
    nuc = '258Fm'

    plot_slices = False
    save_data = True
    save_plt = True
    
    surface_path = os.path.expanduser(f'~/multimodal/surfaces/q20-q30-q40_surfaces/{edf}/{nuc}.dat') 
    path_files = sorted(glob.glob(f'iso-start/258Fm_path_*_Mass_True_Econst_0.0*.txt'))
    
    
    colorArr = ['lime','blue','red','orange','black']
    
    PESdf = pd.read_csv(surface_path,delimiter=',')
    
    #NOTE: THE COORDINATES ARE expected_* IN THE DATA FRAME
    
    uniq_coords = [np.unique(PESdf['expected_q20'].to_numpy()),np.unique(PESdf['expected_q30'].to_numpy()),np.unique(PESdf['expected_q40'].to_numpy())]
    gridDims= [len(uniq_coords[0]),len(uniq_coords[1]),len(uniq_coords[2])]
    
    grids = np.meshgrid(*uniq_coords)
    grids_swap = []
    for m in grids:
        grids_swap.append(np.swapaxes(m,0,1))
        
    EE = PESdf['EHFB'].to_numpy().reshape(*gridDims)
    M22_grid = PESdf['M_22'].to_numpy().reshape(*gridDims)
    M23_grid = PESdf['M_23'].to_numpy().reshape(*gridDims)
    M24_grid = PESdf['M_24'].to_numpy().reshape(*gridDims)
    M33_grid = PESdf['M_33'].to_numpy().reshape(*gridDims)
    M34_grid = PESdf['M_34'].to_numpy().reshape(*gridDims)
    M44_grid = PESdf['M_44'].to_numpy().reshape(*gridDims)
    mass_grids = [M22_grid,M23_grid,M24_grid,M33_grid,M34_grid,M44_grid]
    
    
    ### Find minimum on DFT grids.
    minima_ind = utilities.SurfaceUtils.find_all_local_minimum(EE)
    gs_ind = utilities.SurfaceUtils.find_local_minimum(EE,searchPerc=[0.20,0.20,0.20],returnOnlySmallest=True)
    gs_coord_grid = np.array((grids_swap[0][gs_ind],grids_swap[1][gs_ind],grids_swap[2][gs_ind])).T
    
    iso_ind = utilities.SurfaceUtils.find_local_minimum(EE,searchPerc=[0.50,0.50,0.40],returnOnlySmallest=False)
    iso_coord_grid_arr = np.array((grids_swap[0][iso_ind],grids_swap[1][iso_ind],grids_swap[2][iso_ind])).T
    
    E_gs_grid = EE[gs_ind]
    
    E_iso_grid_arr = EE[iso_ind]
    iso_sorted = np.argsort(E_iso_grid_arr)
    E_iso_grid  = E_iso_grid_arr[iso_sorted[0]]
    iso_coord_grid = iso_coord_grid_arr[iso_sorted[0]]
    #print(iso_sorted)
    #print(E_iso_grid)
    print(f'DFT grid E_gs {E_gs_grid}')
    print(f'DFT grid E_gs coordinate: {gs_coord_grid}')
    
    print(f'DFT grid E_iso {E_iso_grid}')
    print(f'DFT grid E_iso coordinate: {iso_coord_grid}')
    
    shiftE = 0.0
    
    
    EE -= E_iso_grid - shiftE
    
    
    ###############################################################################
    # Plot DFT grids
    ###############################################################################
    if plot_slices == True:
        for i,j in enumerate(range(0,gridDims[2])):
            #print(i)
            fig, ax = plt.subplots(1,1)
            im = ax.pcolormesh(uniq_coords[0],uniq_coords[1],EE[:,:,i].T.clip(0,50),
                               cmap='Spectral_r')
            # im = ax.contourf(uniq_coords[0],uniq_coords[1],EE[:,:,i].T.clip(0,50),levels= 100,
            #                  extend='both',cmap='Spectral_r')
            #cs = ax.contour(xx_s,yy_s,zz_s.clip(-10,20),levels=12,colors='black')
            ax.contour(uniq_coords[0],uniq_coords[1],EE[:,:,i].T.clip(0,50),levels=[0],colors='white',linewidths=2)
            cbar = fig.colorbar(im,ax=ax)
        
            #plt.plot(gs_coord_grid[0],gs_coord_grid[1],'x',color='red',ms=10)
            plt.plot(iso_coord_grid[0],iso_coord_grid[1],'x',color='red',ms=10)
        
            plt.xlabel(r'$Q_{20}$ (b)')
            plt.ylabel(r'$Q_{30}$ (b$^{3/2}$)')
            plt.title(f'Q40 = {uniq_coords[2][i]}')
            plt.show()
            
    ### create the interpolation function using the shifted surface
    V_func = utilities.NDInterpWithBoundary(uniq_coords,EE,splKWargs={'order':3},transformFuncName='smooth_abs')
    
    ###############################################################################
    ## Create inertia tensor functions
    ##############################################################################
    M_func = utilities.PositiveSemidefInterpolator(uniq_coords,mass_grids,_test_nd=False)
    
    
    ###############################################################################
    # Find minimum on finer grid mesh calculated using the interpolation function
    # and define shift function used for NEB calculation
    ###############################################################################
    
    
    if plot_slices == True:
        for i,j in enumerate(range(0,gridDims[2])):
            EE_fine = V_func(np.moveaxis(np.array(grids),0,-1))
            fig, ax = plt.subplots(1,1)
    
            im = ax.contourf(uniq_coords[0],uniq_coords[1],EE[:,:,i].clip(0,50),levels= 100,
                              extend='both',cmap='Spectral_r')
            #cs = ax.contour(xx_s,yy_s,zz_s.clip(-10,20),levels=12,colors='black')
            ax.contour(uniq_coords[0],uniq_coords[1],EE[:,:,i].clip(0,50),levels=[0],colors='white',linewidths=2)
            cbar = fig.colorbar(im,ax=ax)
        
            #plt.plot(gs_coord_grid[0],gs_coord_grid[1],'x',color='red',ms=10)
            plt.plot(iso_coord_grid[0],iso_coord_grid[1],'x',color='red',ms=10)
        
            plt.xlabel(r'$Q_{20}$ (b)')
            plt.ylabel(r'$Q_{30}$ (b$^{3/2}$)')
            plt.title(f'Q40 = {uniq_coords[2][i]}')
            plt.show()

    fig, ax = plt.subplots(1,1)
    im = ax.contourf(uniq_coords[0],uniq_coords[2],EE[:,0].T.clip(0,50),levels= 100,
                      extend='both',cmap='Spectral_r')
    #ax.clabel(cs, cs.levels, inline=True, fontsize=8)
    otl1 = ax.contour(uniq_coords[0],uniq_coords[2],EE[:,0].T.clip(0,50),levels=[0],colors='white',linewidths=1.5)
    

    ax.clabel(otl1, otl1.levels, inline=True, fontsize=10)
    cbar = fig.colorbar(im,ax=ax)

    for i,fpath in enumerate(path_files):        
        print(fpath)
        data = np.loadtxt(fpath,delimiter=',',skiprows=1) 
        print(f'Energy at OTL: {V_func(data[-1])}')
        ax.plot(data[:,0],data[:,2],'--',ms=24,color=colorArr[i], zorder=10, clip_on=False)
        ax.plot(data[:,0][-1],data[:,2][-1],'s',ms=4,color=colorArr[i])
        path_call = utilities.InterpolatedPath(data)
        action = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func,M_func],tfKWargs={})[1][0],4)
        print(f'Action = {action}')
        print(f'Exit point = {data[-1]}')
    

    plt.xlim([120,320])
    plt.ylim([0,200])
    plt.xlabel(r'$Q_{20}$ (b)')
    plt.ylabel(r'$Q_{30}$ (b$^{3/2}$)')
    plt.legend()
    plt.savefig(f'{nuc}_shifts.pdf')
    plt.show()

    for i,fpath in enumerate(path_files):
        
        data = np.loadtxt(fpath,delimiter=',',skiprows=1) 
        energy = V_func(data)
        plt.plot(data[:,0],energy)
        plt.show()
