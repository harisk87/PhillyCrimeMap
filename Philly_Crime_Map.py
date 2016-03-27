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
sum(pd.isnull(philly06['LAT'])) 
sum(pd.isnull(philly06['LONG'])) 

#Remove null values from dataframe 
data = pd.DataFrame(philly06, columns=['LAT','LONG'])
data = data.dropna()

#==============================================================================
# Optimize bandwidth
#==============================================================================
#Create 10,000 datapoint subset of the data to test for best bandwidth (otherwise runtime is very long)
bw_test_data = data.sample(n=10000,replace=False)
features = ['LAT', 'LONG']
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
#best bandwidth first time:  0.00206913808111
#best bandwidth second time: 0.00183298071083
#best bandwidth third time: 0.00183298071083
#==============================================================================
# Compute Kernel Density Estimate
#==============================================================================

# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_  #using defaults: Euclidean distance and Gaussian kernel

#Create data matrix
features = ['LAT', 'LONG']
data_matrix = data.as_matrix(columns = features)
kde.fit(data_matrix)

#==============================================================================
# Set up Plots
#==============================================================================
# Find Max and Min Longitude value

def getMapSizeandImages(dataframe, zoom=14):
    latmax = dataframe['LAT'].max()  #Northmost point : 40.137445
    latmin = dataframe['LAT'].min()  #Southmost point : 39.875032
    lonmax = dataframe['LON'].max()  #Eastmost point : -74.957504
    lonmin = dataframe['LON'].min() #Westmost point : -75.27426
    delta_lat = latmax-latmin
    delta_long = lonmax-lonmin
    lat_deg = latmin
    lon_deg = lonmin
    #
    a, bbox = STT.getImageCluster(lat_deg, lon_deg, delta_lat,  delta_long, zoom)
    return a, bbox
#zoom = 14 shows whole city, higher values = more zoomed 
a, bbox = getMapSizeandImages(data)

#==============================================================================
# Get Plot dimensions and KDE samples for Contour plot
#==============================================================================
xmin = bbox[0]
ymin = bbox[1]
xmax = bbox[2]
ymax = bbox[3]
x = np.arange(xmin, xmax,0.001)
y = np.arange(ymin, ymax, 0.001)
X,Y = np.meshgrid(x,y)
xy = np.vstack([Y.ravel(),X.ravel()]).T
#Get KDE for the points in the plot
Z = np.exp(kde.score_samples(xy))
Z = Z.reshape(X.shape)

#==============================================================================
# Plot Stamen Toner Map Tiles
#==============================================================================
fig = plt.figure(figsize=(12,12)) 
ax = plt.subplot(111)
m = Basemap(
    llcrnrlon=bbox[0], llcrnrlat=bbox[1],
    urcrnrlon=bbox[2], urcrnrlat=bbox[3],
    projection='merc', ax=ax
)
m.imshow(a, interpolation='lanczos', origin='upper')
fig.savefig('PhillyZoom14_map.png', bbox_inches='tight', pad_inches=0)
#==============================================================================
# ##Display points for crime incidences on map tiles (long, lat)
#==============================================================================
#lonlat = zip(philly06.POINT_X.values, philly06.POINT_Y.values)
#ls_points = [m(x,y) for x,y in lonlat]
#ax.scatter([point[0] for point in ls_points],
#          [point[1] for point in ls_points],
#          alpha = 0.9)
#==============================================================================
# Create Contour Plot
#==============================================================================
#Create 'hotness' levels for the contour plot
#maxZ = 211.50162631233624
#maxZ 2nd time = 223.53529395143997  
# Set max level to the highest level of Z at Zoom level = 14 (showing the whole city), so that if we zoom in on a lower-crime area, we won't have the heat levels reset to relative levels
levels = np.linspace(0, Z.max(), 25)

fig = plt.figure(figsize=(12,12))
ax = plt.subplot(111)
plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
fig.savefig('philly2006contour.png', bbox_inches='tight')
plt.show()

#==============================================================================
# Make Contour Plot and Save with no frame/axis (for overlay)
#==============================================================================
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
plt.axis("off")
levels = np.linspace(0, Z.max(), 25)
plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) #gets extent of bbox for just the image
plt.savefig('philly2006contour.png', bbox_inches=extent)

#==============================================================================
# Make Map Plot and Save with no frame/axis (for overlay)
#==============================================================================
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
plt.axis("off")
m = Basemap(
    llcrnrlon=X.min(), llcrnrlat=Y.min(),
    urcrnrlon=X.max(), urcrnrlat=Y.max(),
    projection='merc', ax=ax
)
m.imshow(a, alpha=1, interpolation='lanczos', origin='upper')
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) #gets extent of bbox for just the image
plt.savefig('PhillyZoom14_map.png', bbox_inches=extent)


#==============================================================================
# #Alpha Layer version
#==============================================================================
from PIL import Image
img = Image.open
img = img.convert("RGBA")
datas = img.getdata()

newData = []
for item in datas:
    if item[0] == 255 and item[1] == 255 and item[2] == 255:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)
img.putdata(newData)
masked_img = np.ma.masked_where(img == 0, img)


import matplotlib.image as mpimg
fig = plt.figure(figsize=(12,12))
ax = plt.subplot(111)
extent = xmin, xmax, ymin, ymax
img2=mpimg.imread('philly2006contour.png')
img = mpimg.imread('PhillyZoom14_map.png')
contourlayer = plt.imshow(img2,interpolation="nearest",extent=extent)
plt.hold(True)
maplayer = plt.imshow(img, alpha=.5, interpolation='bilinear',extent=extent)
plt.show()
plt.savefig('PhillyZoom14_alpha.png')


#==============================================================================
# #Plots Basic Line plot version
#==============================================================================

fig, ax = plt.subplots()
for bandwidth in [0.001, 0.01, 0.1]:
    ax.plot(Z,
            label='bw={0}'.format(bandwidth), linewidth=3, alpha=0.5)
            
