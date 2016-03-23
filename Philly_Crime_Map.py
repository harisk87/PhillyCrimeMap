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

#==============================================================================
# Load the Data
#==============================================================================
philly06 = pd.read_csv('GIS_POLICE.INCIDENTS_2006.csv')

#Construct a Kernel Density estimate of the distribution
philly06['Lat'] = philly06['POINT_Y']
philly06['Long'] = philly06['POINT_X']

#Get crimes that aren't theft
philly06 = philly06[philly06['UCR_GENERAL']<=500]
#If we leave theft in, there is a huge concentration in University City/Center City which are the richest/safest parts of the city, we care more about where it is dangerous to be because of crime (and the difference may not even be real, it may be that people in more dangerous parts are less likely to report theft, not that it happens less)
#when we remove theft, count goes from 90,000+ to 32,759

#Look for null values
sum(pd.isnull(philly06['Lat'])) 
sum(pd.isnull(philly06['Long'])) 

#Remove null values from dataframe 
data = pd.DataFrame(philly06, columns=['Lat','Long'])
data = data.dropna()

#==============================================================================
# Optimize bandwidth
#==============================================================================
#Create subset of the data to test for best bandwidth (otherwise runtime is very long)
bw_test_data = data.sample(n=10000,replace=False)
features = ['Lat', 'Long']
bw_test_matrix = bw_test_data.as_matrix(columns = features)

#Test for best bandwidth
params = {'bandwidth': np.logspace(-1., 1., 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(bw_test_matrix)
print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth)) 

#This found that the best estimator was 0.1, since that is the lowest allowed by this set, I will now test a lower range
params = {'bandwidth': np.logspace(-2., -1., 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(bw_test_matrix)
print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth)) 

#Best bandwidth is again the lowest in the range , 0.01, so I will try again with lower bandwidth range
params = {'bandwidth': np.logspace(-3., -2., 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(bw_test_matrix)
print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth)) 
#best bandwidth = 0.00206913808111

#==============================================================================
# Compute Kernel Density Estimate
#==============================================================================

# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_  #using defaults: Euclidean distance and Gaussian kernel

#Create data matrix
features = ['Lat', 'Long']
data_matrix = data.as_matrix(columns = features)
kde.fit(data_matrix)

#==============================================================================
# Set up Plots 
#==============================================================================
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

xmin = bbox[0]
ymin = bbox[1]
xmax = bbox[2]
ymax = bbox[3]
x = np.arange(xmin, xmax,0.001)
y = np.arange(ymin, ymax, 0.001)
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
#m.imshow(a, interpolation='lanczos', origin='upper')
plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
plt.show()
#Next Step: Need to figure out how to plot this on top of the map 

#Plots Basic Line plot version
fig, ax = plt.subplots()
for bandwidth in [0.002, 0.01, 0.1]:
    ax.plot(Z,
            label='bw={0}'.format(bandwidth), linewidth=3, alpha=0.5)