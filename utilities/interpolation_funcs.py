import numpy as np
from scipy import interpolate
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
def rbf_M_func(coord_arrays,M):
    rbf = interpolate.Rbf(coord_arrays[0][:len(coord_arrays[0])-1],coord_arrays[1][:len(coord_arrays[1])-1],M[:len(M)-1])
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
