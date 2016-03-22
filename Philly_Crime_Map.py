# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 19:57:59 2015

@author: heathersimpson
"""

import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import StamenTonerTilesAccess as STT
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

philly06 = pd.read_csv('GIS_POLICE.INCIDENTS_2006.csv')


# Find Max and Min Longitude value
latmax = philly06['POINT_Y'].max()  #Northmost point : 40.137445
latmin = philly06['POINT_Y'].min()  #Southmost point : 39.875032
lonmax = philly06['POINT_X'].max()  #Eastmost point : -74.957504
lonmin = philly06['POINT_X'].min() #Westmost point : -75.27426
lonlat = zip(philly06.POINT_X.values, philly06.POINT_Y.values)
lat_deg = latmin
lon_deg = lonmin
delta_lat = latmax-latmin
delta_long = lonmax-lonmin
zoom = 14

#lat_deg, lon_deg, delta_lat,  delta_long, zoom = lat_deg-delta_lat/2, lon_deg-delta_long/2, delta_lat,  delta_long, 11
#lat_deg, lon_deg, delta_lat,  delta_long, zoom = 45.720-0.04/2, 4.210-0.08/2, 0.04,  0.08, 14
a, bbox = STT.getImageCluster(lat_deg, lon_deg, delta_lat,  delta_long, zoom)

##Display points for crime incidences on map tiles (long, lat)
#ls_points = [m(x,y) for x,y in lonlat]
#ax.scatter([point[0] for point in ls_points],
#          [point[1] for point in ls_points],
#          alpha = 0.9)

#We use a Kernel Density estimate so that we are no t
#Construct a Kernel Density estimate of the distribution
philly06['Lat'] = philly06['POINT_Y']
philly06['Long'] = philly06['POINT_X']

#clean out bad Lat/Long data from philly06['Long'] and ['Lat]
[type(i) for i in philly06['Lat'] if type(i)!=np.float64]
[type(i) for i in philly06['Long'] if type(i)!=np.float64]
#Look for null values
sum(pd.isnull(philly06['Lat'])) #1095
sum(pd.isnull(philly06['Long'])) #1095

#Remove null values from dataframe 
data = pd.DataFrame(philly06, columns=['Lat','Long'])
data = data.dropna()

features = ['Lat', 'Long']
data_matrix = data.as_matrix(columns = features)

#params = {'bandwidth': np.logspace(-1, 1, 20)}
#grid = GridSearchCV(KernelDensity(), params, 2)
#grid.fit(data)

#print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
# use the best estimator to compute the kernel density estimate
#kde = grid.best_estimator_

kde = KernelDensity(bandwidth=0.04, kernel='gaussian')
kde.fit(data_matrix)

#==============================================================================
# Set up Contour Plot
#==============================================================================
xmin = bbox[0]
ymin = bbox[1]
xmax = bbox[2]
ymax = bbox[3]
x = np.arange(xmin, xmax,0.003)
y = np.arange(ymin, ymax, 0.003)
X,Y = np.meshgrid(x,y)
xy = np.vstack([Y.ravel(),X.ravel()]).T

fig = plt.figure(figsize=(10, 10)) 
ax = plt.subplot(111)
Z = np.exp(kde.score_samples(xy))
Z = Z.reshape(X.shape)
levels = np.linspace(0, Z.max(), 25)


m = Basemap(
    llcrnrlon=bbox[0], llcrnrlat=bbox[1],
    urcrnrlon=bbox[2], urcrnrlat=bbox[3],
    projection='merc', ax=ax
)
m.imshow(a, interpolation='lanczos', origin='upper')
plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
plt.show()

fig, ax = plt.subplots()
for bandwidth in [0.1, 0.3, 1.0]:
    ax.plot(Z,
            label='bw={0}'.format(bandwidth), linewidth=3, alpha=0.5)
#ax.hist(Z, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
#ax.set_xlim(-4.5, 3.5)
#ax.legend(loc='upper left')

#ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
#ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")