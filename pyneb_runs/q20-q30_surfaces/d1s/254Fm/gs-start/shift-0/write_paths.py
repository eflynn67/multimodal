import numpy as np
import matplotlib.pyplot as plt
import glob
from datetime import date
import sys
import os
import pandas as pd
### add pyneb
pyneb_path = os.path.expanduser('~/pyneb/src/pyneb') 
utils_path = os.path.expanduser('~/multimodal/utilities')
sys.path.insert(0, pyneb_path)
sys.path.insert(0, utils_path)
import utilities
import utils

if __name__ == "__main__":
    
    today = date.today()
    ### Define nucleus data path (assumes our github structure)
    nuc = '254Fm_D1S'
    edf = 'd1s'

    use_mass = True
    save_data = True
    save_plt = False
    surface_path = os.path.expanduser(f'~/multimodal/surfaces/q20-q30_surfaces/{edf}/{nuc}.h5') 


    ### defines PES object from utils.py
    PES = utils.PES(surface_path)
    uniq_coords = PES.get_unique(return_type='array')
    grids,EE = PES.get_grids(return_coord_grid=True)
    coord_arrays,eng = PES.get_coord_arrays()

    mass_grids = PES.get_mass_grids()
    mass_keys = mass_grids.keys()
    ### IMPORTANT: LIST THE INDICIES OF THE MASS TENSOR TO USE.
    mass_tensor_indicies = ['20','30']


    ### Find minimum on DFT grids.
    minima_ind = utilities.SurfaceUtils.find_all_local_minimum(EE)
    gs_ind = utilities.SurfaceUtils.find_local_minimum(EE,searchPerc=[0.20,0.20],returnOnlySmallest=True)
    gs_coord_grid = np.array((grids[0][gs_ind],grids[1][gs_ind])).T
    
    iso_ind = utilities.SurfaceUtils.find_local_minimum(EE,searchPerc=[0.40,0.40],returnOnlySmallest=False)
    iso_coord_grid_arr = np.array((grids[0][iso_ind],grids[1][iso_ind])).T

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
    fig, ax = plt.subplots(1,1)
    im = ax.contourf(grids[0],grids[1],EE.clip(-5,15),levels= 100,
                     extend='both',cmap='Spectral_r')
    cs = ax.contour(grids[0],grids[1],EE.clip(-5,15),levels=12,colors='black')
    ax.contour(grids[0],grids[1],EE.clip(-5,15),levels=[0],colors='white',linewidths=2)
    cbar = fig.colorbar(im,ax=ax)

    #plt.plot(gs_coord_grid[0],gs_coord_grid[1],'x',color='red',ms=10)
    plt.plot(iso_coord_grid[0],iso_coord_grid[1],'x',color='red',ms=10)

    plt.xlabel(r'$Q_{20}$ (b)')
    plt.ylabel(r'$Q_{30}$ (b$^{3/2}$)')
    plt.title(f'DFT GRIDS {nuc}'+r' SkM$^{*}$')
    plt.show()
############################################
    fig, ax = plt.subplots(1,1)
    im = ax.contourf(grids[0],grids[1],mass_grids['B2020'].clip(0,.05),levels= 100,
                     extend='both',cmap='Spectral_r')
    cbar = fig.colorbar(im,ax=ax)

    #plt.plot(gs_coord_grid[0],gs_coord_grid[1],'x',color='red',ms=10)
    plt.plot(iso_coord_grid[0],iso_coord_grid[1],'x',color='red',ms=10)

    plt.xlabel(r'$Q_{20}$ (b)')
    plt.ylabel(r'$Q_{30}$ (b$^{3/2}$)')
    plt.title(r'DFT $M_{22}$'+f' {nuc}'+r' SkM$^{*}$')
    #plt.savefig('M22_grid_plot.pdf')
    plt.show()


    ### create the interpolation function using the shifted surface
    #V_func_rbf = interpolation_funcs.rbf_V_func(coord_arrays,eng)
    V_func = utilities.NDInterpWithBoundary(uniq_coords,EE,custom_func=None,_test_linear=False,transformFuncName='smooth_abs')
    
    
    
    
    ###############################################################################
    ## Create inertia tensor functions
    ###############################################################################
    #mass_list = {}
    mass_list_psd = []
    #for key in mass_keys:
    #    mass_list[key] = mass_grids[key].reshape(coord_arrays[0].shape)
    #mass_grids_func = {key: rbf_M_func(coord_arrays,mass_list[key],) \
    #              for key in mass_keys}
    for key in mass_keys:
        mass_list_psd.append(mass_grids[key])
    #M_func = utilities.PositiveSemidefInterpolator(uniq_coords,mass_list_psd,_test_nd=False)
    M_func = utilities.PositiveSemidefInterpolator(uniq_coords,mass_list_psd,ndInterpKWargs={'splKWargs':{'kx':1,'ky':1}},_test_nd=False)
    
    ###############################################################################
    # Find minimum on finer grid mesh calculated using the interpolation function
    # and define shift function used for NEB calculation
    ###############################################################################

    uniq_x = np.unique(coord_arrays[0])
    uniq_y = np.unique(coord_arrays[1])
    x_fine = np.linspace(0,uniq_x[-1],800)
    y_fine = np.linspace(0,uniq_y[-1],800)

    xx_fine, yy_fine = np.meshgrid(x_fine,y_fine)
    xy_grid = np.array([xx_fine.flatten(),yy_fine.flatten()]).T
    EE_fine = V_func(xy_grid)
    EE_fine = EE_fine.reshape((800,800))


    minima_ind = utilities.SurfaceUtils.find_all_local_minimum(EE_fine)
    gs_ind = utilities.SurfaceUtils.find_local_minimum(EE_fine,searchPerc=[0.30,0.30],returnOnlySmallest=True)
    gs_coord_fine = np.array((xx_fine[gs_ind],yy_fine[gs_ind])).T
    
    iso_ind_fine = utilities.SurfaceUtils.find_local_minimum(EE_fine,searchPerc=[0.50,0.50],returnOnlySmallest=False)
    iso_coord_fine_arr = np.array((xx_fine[iso_ind_fine],yy_fine[iso_ind_fine])).T
    

    #########

    E_gs_fine = V_func(gs_coord_fine) 
    E_iso_fine_arr = V_func(iso_coord_fine_arr)
    
    iso_sorted = np.argsort(E_iso_fine_arr)
    
    
    E_iso_fine  = E_iso_fine_arr[iso_sorted[0]]
    iso_coord_fine = iso_coord_fine_arr[iso_sorted[0]]
    
    print(f'Fine Grid E_gs: {E_gs_fine}')
    print(f'Fine Grid E_gs Coordinate: {gs_coord_fine}')
    print(f'Fine Grid E_iso: {E_iso_fine}')
    print(f'Fine Grid E_iso Coordinate: {iso_coord_fine}')
    



    ###############################################################################
    # Plot Interpolated grid
    ###############################################################################
    fig, ax = plt.subplots(1,1)
    im = ax.contourf(xx_fine,yy_fine,EE_fine.clip(-5,15),levels= 100,
                     extend='both',cmap='Spectral_r')
    cs = ax.contour(xx_fine,yy_fine,EE_fine.clip(-5,15),levels=12,colors='black')
    ax.contour(xx_fine,yy_fine,EE_fine,levels=[0],colors='white',linewidths=2)
    cbar = fig.colorbar(im,ax=ax)

    #plt.plot(gs_coord_fine[0],gs_coord_fine[1],'x',color='black',ms=10)
    plt.plot(iso_coord_grid[0],iso_coord_grid[1],'x',color='red',ms=10)
    #plt.xlim([0,280])
    #plt.ylim([0,32])
    plt.xlabel(r'$Q_{20}$ (b)')
    plt.ylabel(r'$Q_{30}$ (b$^{3/2}$)')
    plt.title(f'USING interpolator {nuc}'+r' SkM$^{*}$')
    plt.show()
    
    
    paths_dir = glob.glob(f'{nuc}_path_*_Mass_True_Econst_*.txt')
    
    for path_loc in paths_dir:
        path = pd.read_csv(path_loc,sep=',')
        course_energy = V_func(path.to_numpy())
        path_call = utilities.InterpolatedPath(path.to_numpy())
        interp_path,energy_along_path = path_call.compute_along_path(V_func,500,tfArgs=[],tfKWargs={})

        path_loc = path_loc.rstrip('txt')
        inertia_path = M_func(interp_path)
        M22 = []
        M23 = []
        M33 = []
        for M in inertia_path:
            M22.append(M[0][0])
            M23.append(M[0][1])
            M33.append(M[1][1])
        
        data_to_write = np.stack([interp_path[:,0],interp_path[:,1],energy_along_path,M22,M23,M33],axis=-1)
        newDf = pd.DataFrame(data_to_write, columns=['Q20', 'Q30','EHFB','M22','M23','M33'])
        
        newDf.to_csv(path_loc+'dat',sep='\t',index=False)
        
