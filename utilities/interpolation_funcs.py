import numpy as np
from scipy.interpolate import RBFInterpolator
def rbf_V_func(coord_arrays,eng):
    rbf = interpolate.Rbf(coord_arrays[0][:len(coord_arrays[0])-1],coord_arrays[1][:len(coord_arrays[1])-1],eng[:len(eng)-1])
    lb = min(coord_arrays[1])
    def func(points):
        if isinstance(points, np.ndarray) == False:
            points = np.array(points)
        originalShape = points.shape
        if len(originalShape) == 1:
            if points[1] < lb:
                result = 0
            else:
                result = rbf(points[0],points[1])
        elif len(originalShape) == 2:
            for k in range(points.shape[0]):
                if points[k][1] < min(coord_arrays[1]):
                    points[k][1] = 0.0
                else:
                    pass
            result = rbf(points[:,0],points[:,1])
        else:
            raise Exception('Coordinate arrays are wrong shape for interpolator' )
        return result
    return func
def rbf_M_func_2d(uniq_coords,coord_arrays,mass_grids):
    #coord_arrays is assumed to be grid of coordinates in each component 
    # example: coord_arrays[0] should be the 2d corrdinate grid of q20
    # mass_grids is assumed to be dictionary of grid of intertia values for each component of M 
    # each label corresponds to a component of M
    # example: M_arrays[0] is the 2d grid of M22 values
    # the interpolation function takes in rbf(coordinate arry, 1d energy/mass at coordinate arr)
    # for examples [0,0] (coord) -> 10 (value at that coordinate)
    unroll_shape = 0
    coords = np.stack(coord_arrays,axis=-1)
    for p in range(len(uniq_coords)):
        unroll_shape *= len(uniq_coords)
    mass_list = {}
    for key in mass_grids.keys():
        mass_list[key] = mass_grids[key].reshape(unroll_shape)
        mass_grids_func = {key: RBFInterpolator(coords,mass_list[key],) \
                      for key in mass_grids.keys()}
    #interp_func = RBFInterpolator(np.array(df[["q20","q30"]]),df[head])
    lb = min(coord_arrays[1])
    def func(points):
        if isinstance(points, np.ndarray) == False:
            points = np.array(points)
        originalShape = points.shape
        result = np.zeros(points.shape[1],points.shape[1])

        if len(originalShape) == 1:
            if points[1] < lb:
                result[0][0] = 0
                result[1][0] = 0
                result[0][1] = 0
                result[1][1] = 0
            else:
                result[0][0] = mass_grids_func['B2020']
                result[1][0] = 0
                result[0][1] = 0
                result[1][1] = 0
        elif len(originalShape) == 2:
            for k in range(points.shape[0]):
                if points[k][1] < min(coord_arrays[1]):
                    points[k][1] = 0.0
                else:
                    pass
            result = rbf(points[:,0],points[:,1])
        else:
            raise Exception('Coordinate arrays are wrong shape for interpolator' )
        return result
    return func
