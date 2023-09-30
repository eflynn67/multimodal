import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import sys
import os
import time
from scipy import interpolate
import itertools
import pandas as pd
### add pyneb
pyneb_path = os.path.expanduser('~/pyneb/src/pyneb') 
utils_path = os.path.expanduser('~/multimodal/utilities')
sys.path.insert(0, pyneb_path)
sys.path.insert(0, utils_path)
import solvers
import utilities
import utils
import interpolation_funcs

#plt.style.use('science')
fdTol = 10**(-8)

if __name__ == "__main__":
    
    today = date.today()
    ### Define nucleus data path (assumes our github structure)
    edf = 'skms'
    nuc = 'Z_122_N_184'
    
    plot_slices = False
    save_data = False
    save_plt = False

    surface_path = os.path.expanduser(f'~/multimodal/surfaces/q20-q30-q40_surfaces/{edf}/{nuc}.dat') 
    
    PESdf = pd.read_csv(surface_path,delimiter='\s+')
    PESdf = PESdf.sort_values(by=['Q20','Q30','Q40'])

    #NOTE: THE COORDINATES ARE expected_* IN THE DATA FRAME

    uniq_coords = [np.unique(PESdf['Q20'].to_numpy()),np.unique(PESdf['Q30'].to_numpy()),np.unique(PESdf['Q40'].to_numpy())]
    gridDims= [len(uniq_coords[0]),len(uniq_coords[1]),len(uniq_coords[2])]
    
    grids = np.meshgrid(*uniq_coords)
    grids_swap = []
    for m in grids:
        grids_swap.append(np.swapaxes(m,0,1))
        
    EE = PESdf['EHFB'].to_numpy().reshape(*gridDims)
    M22_grid = PESdf['M_22'].to_numpy().reshape(*gridDims)
    M23_grid = PESdf['M_32'].to_numpy().reshape(*gridDims)
    M24_grid = PESdf['M_42'].to_numpy().reshape(*gridDims)
    M33_grid = PESdf['M_33'].to_numpy().reshape(*gridDims)
    M34_grid = PESdf['M_43'].to_numpy().reshape(*gridDims)
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

    
    EE -= E_gs_grid - shiftE


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
    V_func = utilities.NDInterpWithBoundary(uniq_coords,EE,splKWargs={'order':3})
    
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
    
            im = ax.contourf(uniq_coords[0],uniq_coords[1],EE_fine[:,:,i].clip(0,50),levels= 100,
                              extend='both',cmap='Spectral_r')
            #cs = ax.contour(xx_s,yy_s,zz_s.clip(-10,20),levels=12,colors='black')
            ax.contour(uniq_coords[0],uniq_coords[1],EE_fine[:,:,i].clip(0,50),levels=[0],colors='white',linewidths=2)
            cbar = fig.colorbar(im,ax=ax)
        
            #plt.plot(gs_coord_grid[0],gs_coord_grid[1],'x',color='red',ms=10)
            plt.plot(iso_coord_grid[0],iso_coord_grid[1],'x',color='red',ms=10)
        
            plt.xlabel(r'$Q_{20}$ (b)')
            plt.ylabel(r'$Q_{30}$ (b$^{3/2}$)')
            plt.title(f'Q40 = {uniq_coords[2][i]}')
            plt.show()
    
    x_fine = np.linspace(0,uniq_coords[0][-1],205)
    y_fine = np.linspace(0,uniq_coords[1][-1],200)
    z_fine =  np.linspace(0,uniq_coords[2][-1],201)
    yy_fine, xx_fine,zz_fine = np.meshgrid(y_fine,x_fine,z_fine)
    grids_fine = [xx_fine,yy_fine,zz_fine]
    #print(grids[0])
    #print(70*'=')
    #print(xx_fine)

    xyz_grid = np.array([xx_fine.flatten(),yy_fine.flatten(),zz_fine.flatten()]).T
    EE_fine = V_func(xyz_grid)
    
    EE_fine_grid = EE_fine.reshape(len(x_fine),len(y_fine),len(z_fine))
    
    fig, ax = plt.subplots(1,1)
    im = ax.contourf(EE_fine_grid[0].T.clip(0,50),levels= 100,
                     extend='both',cmap='Spectral_r')
    cbar = fig.colorbar(im,ax=ax)
    plt.show()
##################################################################################### 
    '''
    data_dict = {'q20':xyz_grid[:,0],'q30':xyz_grid[:,1],'q40':xyz_grid[:,2],'EHFB':EE_fine}   
    df = pd.DataFrame(data_dict)
    const_names = ['q40']
    plane_names = ['q20','q30','EHFB']
    print(z_fine)
    const_comps = [44]
    print(df)
    for i,key in enumerate(const_names):
        subspace = df.loc[df[key]==const_comps[i]]
    x = subspace[plane_names[0]]
    y = subspace[plane_names[1]]
    V = subspace[plane_names[2]]
    df2 = pd.DataFrame({'x':x,'y':y,'z':V})
    x1 = np.sort(df2.x.unique())
    x2 = np.sort(df2.y.unique())
    xx,yy = np.meshgrid(x1,x2)
    zz = pd.DataFrame(None, index = x1, columns = x2, dtype = float)
    for i, r in df2.iterrows():
        zz.loc[r.x, r.y] = np.around(r.z,3)
    zz = zz.to_numpy()
    zz = zz.T
    
    print(xx.shape)
    print(yy.shape)
    print(zz.shape)
    fig, ax = plt.subplots(1,1)
    im = ax.contourf(xx,yy,zz.clip(-5,50),levels= 100,
                     extend='both',cmap='Spectral_r')
    cs = ax.contour(xx,yy,zz.clip(-5,50),levels=12,colors='black')
    ax.contour(xx.T,yy.T,zz.clip(-5,20).T,levels=[0],colors='white',linewidths=2)
    cbar = fig.colorbar(im,ax=ax)
    plt.show()
    
    '''
    OTS_coords = utilities.SurfaceUtils.find_approximate_contours(grids_fine,EE_fine_grid,eneg=0.3,show=True,
                                  returnAsArr=True)
    OTS_df = pd.DataFrame({'Q20':OTS_coords[:,0],'Q30':OTS_coords[:,1],'Q40':OTS_coords[:,2]})
    OTS_df = OTS_df.drop(OTS_df[OTS_df['Q40']>180].index)
    #OTS_df = OTS_df.drop(OTS_df[OTS_df['Q40']<20].index)
    #OTS_df = OTS_df.drop(OTS_df[OTS_df['Q20']<100].index)
    OTS_df.to_csv(f'{nuc}_OTS_interp_cubic.txt',sep=',',index=False,float_format='%.2f')