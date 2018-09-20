import pandas as pd
import numpy as np
import pylab as pyl
import scipy as scp 
import control as ctrl #https://python-control.readthedocs.io/en/latest/control.html

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.cm as cm

from itertools import repeat
from multiprocessing import Pool
import concurrent.futures

#%%

def calc_road_min_distance(i, j):
    distMin = np.array([np.linalg.norm( [roadGridX[j,i], roadGridY[j,i]] - roadPoints_fine[k,:] ) for k in range(roadPoints_fine.shape[0])])
    if distMin.min() <= roadWidth/2:
        return roadPoints_dist[distMin.argmin(),0]
    else:
        return np.nan

        

roadWidth = 10

roadPoints_qual = 0.25
roadPoints= np.array([[00.00, 00.00],
                      [50.00, 00.00],
                      [53.00, 00.41],
                      [55.92, 01.88],
                      [58.47, 04.59],
                      [59.88, 08.42],
                      [59.91, 11.42],
                      [58.57, 15.18],
                      [56.02, 18.00],
                      [52.82, 19.57],
                      [49.97, 20.00],
                      [47.72, 20.27],
                      [45.26, 21.20],
                      [43.15, 22.69],
                      [41.45, 24.80],
                      [40.37, 27.33],
                      [40.01, 30.34],
                      [40.61, 33.43],
                      [42.45, 36.55],
                      [44.71, 38.48],
                      [47.21, 39.60],
                      [50.00, 40.00],
                      [80.00, 40.00],
                      [81.85, 40.36],
                      [83.97, 41.94],
                      [85.00, 44.79],
                      [84.30, 47.49],
                      [64.32, 82.15],
                      [59.67, 88.00],
                      [54.52, 91.68],
                      [49.68, 93.60],
                      [44.44, 94.55],
                      [39.54, 94.41],
                      [34.61, 93.28],
                      [29.44, 90.84],
                      [25.53, 87.86],
                      [22.21, 84.00],
                      [19.38, 78.69],
                      [18.07, 74.17],
                      [17.69, 70.91],
                      [17.82, 66.80],
                      [18.59, 63.00],
                      [19.41, 60.52],
                      [21.03, 57.13],
                      [23.22, 52.45],
                      [24.15, 49.31],
                      [24.83, 45.54],
                      [25.03, 40.84],
                      [24.44, 36.25],
                      [22.75, 30.67],
                      [19.56, 24.86],
                      [13.96, 18.87],
                      [09.60, 15.91],
                      [05.30, 13.99]])
roadPoints[:,1] = roadPoints[:,1] + 5

roadPoints_fine = roadPoints[0,:]
roadPoints_dist = np.array([0])
for i in np.arange(0,roadPoints.shape[0]-1):
    vec = roadPoints[i+1,:] - roadPoints[i,:]
    vecNorm = np.linalg.norm(vec)
    vecUnit = vec/vecNorm
    for j in np.arange(roadPoints_qual, vecNorm+roadPoints_qual, roadPoints_qual):
        roadPoints_fine = np.vstack( (roadPoints_fine, roadPoints[i,:] + j*vecUnit) ) 
        roadPoints_dist = np.vstack( (roadPoints_dist, roadPoints_dist[-1] + roadPoints_qual) )
    roadPoints_fine = np.vstack( (roadPoints_fine, roadPoints[i+1,:]) )
    roadPoints_dist = np.vstack( (roadPoints_dist, roadPoints_dist[-1] + roadPoints_qual) )

#%%

# s = array specifying meshgrid boundaries
s = np.array([np.arange(0, max(roadPoints[:,0])+roadWidth + roadPoints_qual*2, roadPoints_qual*2),
              np.arange(0, max(roadPoints[:,1])+roadWidth + roadPoints_qual*2, roadPoints_qual*2)])

roadGridX, roadGridY = np.meshgrid( s[0], s[1] )
roadGridZ = np.nan * np.ones(shape=roadGridX.shape)

for i in range(s[0].shape[0]):
    for j in range(s[1].shape[0]):
        distMin = np.array([np.linalg.norm( [roadGridX[j,i], roadGridY[j,i]] - roadPoints_fine[k,:] ) for k in range(roadPoints_fine.shape[0])])
        if distMin.min() <= roadWidth/2:
            roadGridZ[j,i] = roadPoints_dist[distMin.argmin()]
    print('--- {0:3.1f}%'.format(i/s[0].shape[0]*100))



#if __name__ == '__main__': 
#    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
#    for i in range(s[0].shape[0]):
#        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
#            j = range(s[1].shape[0])
#            for j_out, dist_out in zip(j, pool.map( calc_road_min_distance, repeat(i), j )):
#                roadGridZ[j_out,i] = dist_out
#        with Pool(10) as pool:        
#            j = range(s[1].shape[0])
#            for j_out, dist_out in zip(j, pool.starmap( calc_road_min_distance, zip(repeat(i),j) )):
#                roadGridZ[j_out,i] = dist_out
#        print('--- {0:3.1f}%'.format(i/s[0].shape[0]*100))


#%%
    
################# MATPLOTLIB #################
#   
fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'}) 
surf = ax.plot_surface(roadGridX, roadGridY, roadGridZ, rstride=3, cstride=3, edgecolors='m', linewidth=0, antialiased=False, vmin=np.nanmin(roadGridZ), vmax=np.nanmax(roadGridZ)) #cmap=cm.jet,
cb = fig.colorbar(surf)
fig.tight_layout()
plt.show()


#%%

from class_vehicle import class_vehicle

vehicle = class_vehicle(m=1500, Izz=3000, lf=1.3, lr=1.7, cf=25000, cr=40000, roadGridX=roadGridX, roadGridY=roadGridY, roadGridZ=roadGridZ)
vehicle.sim(s0=(10,10), v0=(0,0), psi0=0, psip0=0, beta0=0)