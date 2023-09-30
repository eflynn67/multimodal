import numpy as np
import matplotlib.pyplot as plt

from datetime import date
import sys
import os
import time
from scipy import interpolate
import itertools
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
    nuc = '258Fm'
    edf = 'skmstar'

    save_data = True
    save_plt = True
    
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
    gs_ind = utilities.SurfaceUtils.find_local_minimum(EE,searchPerc=[0.15,0.15],returnOnlySmallest=True)
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

    plt.plot(gs_coord_grid[0],gs_coord_grid[1],'x',color='black',ms=10)
    plt.plot(iso_coord_grid[0],iso_coord_grid[1],'x',color='red',ms=10)
    plt.xlim([0,300])
    plt.ylim([0,32])
    plt.xlabel(r'$Q_{20}$ (b)')
    plt.ylabel(r'$Q_{30}$ (b$^{3/2}$)')
    plt.title(f'DFT GRIDS {nuc}'+r' SkM$^{*}$')
    plt.show()
############################################
    fig, ax = plt.subplots(1,1)
    im = ax.contourf(grids[0],grids[1],mass_grids['B2020'].clip(0,.05),levels= 100,
                     extend='both',cmap='Spectral_r')
    cbar = fig.colorbar(im,ax=ax)

    plt.plot(gs_coord_grid[0],gs_coord_grid[1],'x',color='black',ms=10)
    plt.plot(iso_coord_grid[0],iso_coord_grid[1],'x',color='red',ms=10)

    plt.xlabel(r'$Q_{20}$ (b)')
    plt.ylabel(r'$Q_{30}$ (b$^{3/2}$)')
    plt.title(r'DFT $M_{22}$'+f' {nuc}'+r' SkM$^{*}$')
    plt.xlim([0,300])
    plt.ylim([0,32])
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
    M_func = utilities.PositiveSemidefInterpolator(uniq_coords,mass_list_psd,ndInterpKWargs={'_test_linear':True},_test_nd=False)

    
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
    gs_ind = utilities.SurfaceUtils.find_local_minimum(EE_fine,searchPerc=[0.20,0.20],returnOnlySmallest=True)
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

    plt.plot(gs_coord_fine[0],gs_coord_fine[1],'x',color='black',ms=10)
    plt.plot(iso_coord_grid[0],iso_coord_grid[1],'x',color='red',ms=10)
    plt.xlim([0,300])
    plt.ylim([0,32])
    plt.xlabel(r'$Q_{20}$ (b)')
    plt.ylabel(r'$Q_{30}$ (b$^{3/2}$)')
    plt.title(f'USING interpolator {nuc}'+r' SkM$^{*}$')
    plt.show()


###############################################################################
## NEB Calculation
###############################################################################
    
    path_type = ['Asymmetric','Symmetric_1','Symmetric_2']
    NImgs = 82
    k = 15.0 
    kappa = 2.0
    
    E_const =  0
    V_func_shift = V_func
    nDims = len(uniq_coords)
    force_R0 = True
    force_RN = True
    springR0 = False
    springRN = True
    
    
    endPointFix = (force_R0,force_RN)
    springForceFix = (springR0,springRN)
    
    
    dt = .1
    NIterations_const =0
    NIterations_var =5000
    
    ### define initial path
    R0 = gs_coord_grid
    RNArr = [[100.27,0]]#[100.27,0]]
    #RNArr = [[132,0],[132,0]]
    colorArr = ['purple','red','lime']
    
    LAPArr_const = []
    actionArr_const = []
    finalAction_const = [] # storing the final action for file write
    init_array = []
    ###############################################################################
    ########### Run with constant inertia
    ###############################################################################
    if NIterations_const > 0:
        for p,RN in enumerate(RNArr):
    
            ### Optimization parameters
            FireParams = {"dtMax":.1,"dtMin":10**(-6),"nAccel":10,"fInc":1.1,"fAlpha":0.99,\
                 "fDecel":0.5,"aStart":0.1,"maxmove":np.array([.1,.1])}
            #### target function definition
            target_func_LAP = utilities.TargetFunctions.action_squared
            target_func_grad_LAP = utilities.GradientApproximations().discrete_sqr_action_grad
            
            
            init_path_constructor = utils.init_NEB_path(R0,RN,NImgs)
            if p == 0:
                init_path = init_path_constructor.deform()
            else:
                init_path = init_path_constructor.linear_path()
            init_array.append(init_path)
            ### Define parameter dictionaries (mostly for book keeping)
            neb_params ={'k':k,'kappa':kappa,'constraintEneg':E_const}
    
    
            #### Compute LAP
            # LAP function you want to minimize
    
            LAP_params = {'potential':V_func_shift,'nPts':NImgs,'nDims':nDims,'mass':None,'endpointSpringForce': springForceFix ,\
                             'endpointHarmonicForce':endPointFix,'target_func':target_func_LAP,\
                             'target_func_grad':target_func_grad_LAP,'nebParams':neb_params,'logLevel':0}
    
            ### define the least action object
            ### This essentially defines the forces given the target and gradient functions
            lap = solvers.LeastActionPath(**LAP_params)
    
            ### Define the optimizer object to use. Note the initial band is passed
            ### here and the operations defined in LeastActionPath are applied to
            ### the band.
            minObj_LAP = solvers.VerletMinimization(lap,initialPoints=init_path)
    
            ### Begining the optimization procedure. Results are all of the velocities
            ### band positions, and forces for each iteration of the optimization.
            t0 = time.time()
            #tStepArr, alphaArr, stepsSinceReset,dummy = minObj_LAP.fire2(dt,NIterations_const,fireParams=FireParams,useLocal=False)
            tStepArr, alphaArr, stepsSinceReset, dummy = minObj_LAP.fire(dt,NIterations_const,fireParams=FireParams,useLocal=False,earlyStop=False)
            allPaths_LAP = minObj_LAP.allPts
    
            final_path_LAP = allPaths_LAP[-1]
            t1 = time.time()
            total_time_LAP = t1 - t0
            
            print('total_time LAP: ',total_time_LAP)
            action_array_LAP = np.zeros(NIterations_const+2)
    
            # function returns matrix of functions
            for i,path in enumerate(allPaths_LAP):
                #path_call = utilities.InterpolatedPath(path)
                #action_array_LAP[i] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func_shift],tfKWargs={})[1][0],4)
                action_array_LAP[i] = utilities.TargetFunctions.action(path, V_func_shift)[0]
            min_action_LAP = np.around(action_array_LAP[-1],2)
            finalAction_const.append(min_action_LAP)
            LAPArr_const.append(final_path_LAP)
            actionArr_const.append(action_array_LAP)
             
            E_on_path = V_func(final_path_LAP)
            plt.plot(np.linspace(0,1,len(final_path_LAP)),E_on_path)
            plt.title(f'E on path {path_type[p]}')
            plt.show()
         
        ### plot constant inertia trajectories 
        fig, ax = plt.subplots(1,1)
        im = ax.contourf(xx_fine,yy_fine,EE_fine.clip(-5,15),levels= 100,extend='both',cmap='Spectral_r')
        cs = ax.contour(xx_fine,yy_fine,EE_fine.clip(-5,15),levels=6,colors='black')
        ax.contour(xx_fine,yy_fine,EE_fine,levels=[E_const],colors='white',linewidths=2)
        cbar = fig.colorbar(im)
        cbar.ax.tick_params()
        ax.set_ylabel('$Q_{30}$')
        ax.set_xlabel('$Q_{20}$')
        for k in range(len(LAPArr_const)):
            ax.plot(LAPArr_const[k][:, 0], LAPArr_const[k][:, 1], '.-',ms=4,label=f'{path_type[k]}',color=colorArr[k],linewidth=3)
            ax.plot(init_array[k][:, 0], init_array[k][:,1], '-',ms=12,label='initial',color='orange',linewidth=3) 
            print(f"Constant Inertia Entrance Point: {LAPArr_const[k][0]}")
            print(f"Constant Inertia Entrance Energy: {V_func_shift(LAPArr_const[k][0])[0]}" )
            print(f"Constant Inertia Exit Point: {LAPArr_const[k][-1]}")
            print(f"Constant Inertia Exit Energy: {V_func_shift(LAPArr_const[k][-1])[0]}" )
            
        plt.title(f'{nuc} Constant inertia tensor')
        plt.legend()
        plt.xlim([0,280])
        plt.ylim([0,32])
        plt.show()
        ### Plot action for the constant inertia trajectories    
        
        fig2, ax2 = plt.subplots(1,1)
        for k in range(len(actionArr_const)):
            ax2.plot(range(NIterations_const+2),actionArr_const[k],label=f'{path_type[k]}: '+str(np.round(actionArr_const[k][-1],2)),color=colorArr[k])
            ax2.set_xlabel('Iterations')
            ax2.set_ylabel('Action')
            ax2.legend(frameon=True, fancybox=True)
        plt.title(f'{nuc} Constant inertia tensor Econst = {E_const}')
        plt.legend()
        plt.show()
    
    ###############################################################################
    ########### Now run with inertia using the constant inertia solution as intitial guess.
    ###############################################################################
    LAPArr_var = []
    actionArr_var = []
    finalAction_var = [] 
    for p,RN in enumerate(RNArr):
        if NIterations_const == 0:
            init_path_constructor = utils.init_NEB_path(R0,RN,NImgs)
            init_path = init_path_constructor.linear_path()
            if p == 0:
                init_path = init_path_constructor.deform()
            else: 
                init_path = init_path_constructor.linear_path()
        else:
            init_path = LAPArr_const[p]
        FireParams = {"dtMax":.05,"dtMin":10**(-6),"nAccel":10,"fInc":1.1,"fAlpha":0.99,\
             "fDecel":0.5,"aStart":0.1,"maxmove":np.array([.1,.1])} 
            
        neb_params ={'k':k,'kappa':kappa,'constraintEneg':E_const}


        target_func_LAP = utilities.TargetFunctions.action_squared
        target_func_grad_LAP = utilities.GradientApproximations().discrete_sqr_action_grad

        #### Compute LAP
        # LAP function you want to minimize

        LAP_params = {'potential':V_func_shift,'nPts':NImgs,'nDims':nDims,'mass':M_func,'endpointSpringForce': springForceFix ,\
                         'endpointHarmonicForce':endPointFix,'target_func':target_func_LAP,\
                         'target_func_grad':target_func_grad_LAP,'nebParams':neb_params,'logLevel':0}

        ### define the least action object
        ### This essentially defines the forces given the target and gradient functions
        lap = solvers.LeastActionPath(**LAP_params)

        ### Define the optimizer object to use. Note the initial band is passed
        ### here and the operations defined in LeastActionPath are applied to
        ### the band.
        minObj_LAP = solvers.VerletMinimization(lap,initialPoints=init_path)

        ### Begining the optimization procedure. Results are all of the velocities
        ### band positions, and forces for each iteration of the optimization.
        t0 = time.time()
        #tStepArr, alphaArr, stepsSinceReset,dummy = minObj_LAP.fire2(dt,NIterations,fireParams=FireParams,useLocal=False,earlyStop=False)
        #cProfile.run('minObj_LAP.fire(dt,NIterations,fireParams=FireParams,useLocal=True,earlyStop=False)')
        tStepArr, alphaArr, stepsSinceReset, dummy = minObj_LAP.fire(dt,NIterations_var,fireParams=FireParams,useLocal=False,earlyStop=False)
        
        allPaths_LAP = minObj_LAP.allPts

        final_path_LAP = allPaths_LAP[-1]
        t1 = time.time()
        total_time_LAP = t1 - t0
        print('total_time LAP: ',total_time_LAP)
        action_array_LAP = np.zeros(NIterations_var+2)

        for i,path in enumerate(allPaths_LAP):
            #path_call = utilities.InterpolatedPath(path)
            #action_array_LAP[i] = np.around(path_call.compute_along_path(utilities.TargetFunctions.action,500,tfArgs=[V_func_shift,M_func],tfKWargs={})[1][0],4)
            action_array_LAP[i] = utilities.TargetFunctions.action(path, V_func_shift,M_func)[0]
        min_action_LAP = np.around(action_array_LAP[-1],2)
        finalAction_var.append(min_action_LAP)
        LAPArr_var.append(final_path_LAP)
        actionArr_var.append(action_array_LAP) 
        
        mass_on_path = M_func(final_path_LAP)
        m22_path = []
        m33_path = []
        m23_path = []
        for pnt in mass_on_path:
            m22_path.append(pnt[0][0])
            m33_path.append(pnt[1][1])
            m23_path.append(pnt[0][1])
        param_arr = np.linspace(0,1,len(final_path_LAP))
        plt.plot(param_arr,m22_path,label='M_22')
        plt.plot(param_arr,m23_path ,label='M_23')
        plt.plot(param_arr,m33_path ,label='M_33')
        plt.legend()
        plt.title(f'{nuc} Inertia {path_type[p]}')
        if save_plt == True:
            plt.savefig(f'{nuc} inertia_on_path_{path_type[p]}.pdf')
        plt.show()
        
        E_on_path = V_func(final_path_LAP)

        plt.plot(np.linspace(0,1,len(final_path_LAP)),E_on_path)
        plt.title(f'E on path {path_type[p]}')
        if save_plt == True:
            plt.savefig(f'E_on_path_{path_type[p]}_Econst_{E_const}.pdf')
        plt.show()
    if save_data == True:
        for i in range(len(finalAction_var)):
            np.savetxt(nuc+f'_path_{path_type[i]}_Mass_True_Econst_{E_const}.txt',LAPArr_var[i],comments='',delimiter=',',header="Q20,Q30")
    ### Plot the results.
    
    fig, ax = plt.subplots(1,1)
    im = ax.contourf(grids[0],grids[1],EE.clip(-5,15),levels= 100,extend='both',cmap='Spectral_r')
    cs = ax.contour(grids[0],grids[1],EE.clip(-5,15),levels=6,colors='black')
    ax.contour(grids[0],grids[1],EE,levels=[E_const],colors='white',linewidths=2)
    cbar = fig.colorbar(im)
    cbar.ax.tick_params()
    ax.set_ylabel('$Q_{30}$')
    ax.set_xlabel('$Q_{20}$')
    for k in range(len(LAPArr_var)):
        #ax.plot(LAPArr_const[k][:, 0], LAPArr_const[k][:, 1], '.-',ms=8,color='red',label='Constant')
        ax.plot(LAPArr_var[k][:, 0], LAPArr_var[k][:, 1], '.-',ms=4,label=f'{path_type[k]}',color=colorArr[k],linewidth=3)
        #ax.plot(LAPArr_const[k][:, 0], LAPArr_const[k][:, 1], '.-',ms=8,color='red',label='Constant')
        print(f"Variable Inertia Entrance Point: {LAPArr_var[k][0]}")
        print(f"Variable Inertia Entrance Energy: {V_func_shift(LAPArr_var[k][0])[0]}" )
        print(f"Variable Inertia Exit Point: {LAPArr_var[k][-1]}")
        print(f"Variable Inertia Exit Energy: {V_func_shift(LAPArr_var[k][-1])[0]}" )
    plt.legend()
    plt.xlim([0,280])
    plt.ylim([0,32])
    plt.title(f'{nuc} vari inertia Econst = {E_const}')
    if save_plt == True:
        plt.savefig(f'paths_inertia_Econst_{E_const}.pdf')
    plt.show()
    
    
    fig2, ax2 = plt.subplots(1,1)
    for k in range(len(LAPArr_var)):
        ax2.plot(range(NIterations_var+2),actionArr_var[k],label=f'{path_type[k]}: '+str(np.round(actionArr_var[k][-1],2)),color=colorArr[k])
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Action')
        ax2.legend(frameon=True, fancybox=True)
    
    plt.title(f'{nuc} Variable inertia tensor')
    if save_plt == True:
        plt.savefig(f'action_inertia_Econst_{E_const}.pdf')
    plt.show()