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
import solvers
import utilities
import utils



#plt.style.use('science')
fdTol = 10**(-8)

if __name__ == "__main__":
    
    today = date.today()
    ### Define nucleus data path (assumes our github structure)
    nuc = '256Fm'
    edf = 'skmstar'

    use_mass = True
    save_data = True
    save_plt = False
    surface_path = os.path.expanduser(f'~/multimodal/surfaces/q20-q30_surfaces/{edf}/{nuc}.h5') 
    E_const_arr = [0,0.5,1]
    path_files_asym = sorted(glob.glob('shift-*/256Fm_path_Asymmetric_Mass_True_Econst_0.0*.txt'))
    path_files_sym = sorted(glob.glob('shift-*/256Fm_path_Symmetric_1_Mass_True_Econst_*.txt'))
    print(path_files_asym)
    
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

    
    EE -= E_iso_grid - shiftE
    
    V_func = utilities.NDInterpWithBoundary(uniq_coords,EE,custom_func=None,_test_linear=False)
    
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
    

    ###############################################################################
    # Plot Interpolated grid
    ###############################################################################
    fig, ax = plt.subplots(1,1)
    im = ax.contourf(xx_fine,yy_fine,EE_fine.clip(-5,15),levels= 100,
                     extend='both',cmap='Spectral_r')
    cs = ax.contour(xx_fine,yy_fine,EE_fine.clip(-5,15),levels=[-3,-2,2,4,5],colors='black',linewidths=.8)
    color_arr = ['red','green','blue']
    ax.clabel(cs, cs.levels, inline=True, fontsize=8)
    otl1 = ax.contour(xx_fine,yy_fine,EE_fine,levels=[0],colors=color_arr[0],linewidths=1.5)
    otl2 =ax.contour(xx_fine,yy_fine,EE_fine,levels=[.5],colors=color_arr[1],linewidths=1.5)
    otl3 = ax.contour(xx_fine,yy_fine,EE_fine,levels=[1],colors=color_arr[2],linewidths=1.5)

    ax.clabel(otl1, otl1.levels, inline=True, fontsize=10)
    ax.clabel(otl2, otl2.levels, inline=True, fontsize=10)
    ax.clabel(otl3, otl3.levels, inline=True, fontsize=10)
    cbar = fig.colorbar(im,ax=ax)

    Eshifts = [0.0,0.5,1.0]
    for i,fpath in enumerate(path_files_asym):
        print(fpath)
        data = np.loadtxt(fpath,delimiter=',',skiprows=1) 
        ax.plot(data[:,0],data[:,1],'--',ms=24,color=color_arr[i], zorder=10, clip_on=False)
        ax.plot(data[:,0][-1],data[:,1][-1],'s',ms=4,color=color_arr[i])
        path_call = utilities.InterpolatedPath(data)
        action = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func,M_func],tfKWargs={})[1][0],4)
        print(f'Action = {action}')
        print(f'Exit point = {data[-1]}')

    for i,fpath in enumerate(path_files_sym):
        print(fpath)
        data = np.loadtxt(fpath,delimiter=',',skiprows=1) 
        ax.plot(data[:,0],data[:,1],'-',ms=24,color=color_arr[i], zorder=10, clip_on=False,label=r'$\Delta E$ ='+f'{Eshifts[i]}')
        ax.plot(data[:,0][-1],data[:,1][-1],'s',ms=4,color=color_arr[i],zorder=10, clip_on=False)
        path_call = utilities.InterpolatedPath(data)
        action = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func,M_func],tfKWargs={})[1][0],4)
        print(f'Action = {action}')
        print(f'Exit point = {data[-1]}')

    plt.xlim([100,280])
    plt.ylim([0,28])
    plt.xlabel(r'$Q_{20}$ (b)')
    plt.ylabel(r'$Q_{30}$ (b$^{3/2}$)')
    plt.legend()
    plt.savefig(f'{nuc}_shifts.pdf')
    plt.show()
    
    