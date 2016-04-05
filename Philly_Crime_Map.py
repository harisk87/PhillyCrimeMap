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
from matplotlib.widgets import Slider, Button, RadioButtons

#==============================================================================
# Get Philly Map Tiles
#==============================================================================
# Find Max and Min Longitude value
zoom = 12
def getMapSizeandImages(zoom=12): 
    #using the 2006 crime data LAT and LONG max/min as default values for zoom = 14
    latmax = 40.137445  #Northmost point : 40.137445
    latmin = 39.875032  #Southmost point : 39.875032
    lonmax = -74.957504 #Eastmost point : -74.957504
    lonmin = -75.27426  #Westmost point : -75.27426
    delta_lat = latmax-latmin
    delta_long = lonmax-lonmin
    lat_deg = latmin
    lon_deg = lonmin
    a, bbox = STT.getImageCluster(lat_deg, lon_deg, delta_lat*.9,  delta_long*.9, zoom)
    return a, bbox
#zoom = 14 shows whole city, higher values = more zoomed 
a, bbox = getMapSizeandImages()

#==============================================================================
# Get Plot dimensions for Map and Contour
#==============================================================================

xmin = bbox[0]
ymin = bbox[1]
xmax = bbox[2]
ymax = bbox[3]
x = np.arange(xmin, xmax,0.001)
y = np.arange(ymin, ymax, 0.001)
X,Y = np.meshgrid(x,y)
xy = np.vstack([Y.ravel(),X.ravel()]).T
#==============================================================================
# Plot StamenToner Map Tiles and Save Image (for overlay)
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
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())  #gets extent of bbox for just the plot (without the frame/padding), so we can save just the plot extent (no frame / axis), allowing us to overlay the images later    
plotname = 'philly_zoom%s_map.png' %(zoom)
plt.savefig(plotname, bbox_inches=extent) #save image of plot

#==============================================================================
# Load the Crime Data
#==============================================================================

#years = ['2006', '2007']

best_bandwidths = []  #We'll keep track of the best bandwidth parameters found for each model by GridSearchCV so we can make sure they all fell within the range tested (results at the lower and upper bounds would indicate that we should retest with a lower/higher range)
max_crime_density = [] # Keep track of maximum density estimate for each model (max Z), and use that to set the heatmap levels

#for year in years:

#There was some bad data in Incidents_2007 file, it refused to load in pd.read_csv but would hang indefinitely, I found non-unicode chars in there 
#I will assume that it may happen for other files also, so will clean all  
year = '2007'
crimefile = 'CrimeData/Incidents_%s.csv' % (year)
f = open(crimefile, 'r')
lines = f.readlines() 
f.close()
lines[0]# This shows that the first row (header row) contains some weird characters \xef\xbb\xbf, so I'll clean those out
#First I tried ignoring the header row,but there are still other weird chars in the file

cleaned_file = 'CrimeData/Incidents_%s_cleaned.csv' %(year)
#erase file contents if they already exist so that we don't keep appending data to the same file
f = open(cleaned_file, 'w') 
f.write('')
f.close
f = open(cleaned_file, 'a')
for l in lines:
    l = unicode(l, errors='ignore') #will leave out any chars that can't be converted to unicode
    f.write(l)
crime_colnames = "X,Y,DC_DIST,SECTOR,DISPATCH_DATE_TIME,DISPATCH_DATE,DISPATCH_TIME,HOUR,DC_KEY,LOCATION_BLOCK,UCR_GENERAL,OBJECTID,TEXT_GENERAL_CODE,POINT_X,POINT_Y,GlobalID"
crime_colnames = crime_colnames.split(",")
crime_dtypes = [float, float, int, str, str,str,str,str,float,str,int,int,str,float,float]
crime_dtypedict = dict(zip(crime_colnames,crime_dtypes))
phillycrime = pd.read_csv(cleaned_file,names=crime_colnames, dtype= crime_dtypedict, skiprows=[0], sep=",", quotechar="\"", encoding='utf-8')

#Construct a Kernel Density estimate of the distribution
phillycrime['LAT'] = phillycrime['POINT_Y']
phillycrime['LONG'] = phillycrime['POINT_X']

#Get crimes that aren't theft
phillycrime = phillycrime[phillycrime['UCR_GENERAL']<=500] #If we leave theft in, there is a huge concentration in University City/Center City which are the richest/safest parts of the city, we care more about where it is dangerous to be because of crime (and the difference may not even be real, it may be that people in more dangerous parts are less likely to report theft, not that it happens less) 
#when we remove theft, count goes from 90,000+ to 32,759

#Look for null values
sum(pd.isnull(phillycrime['LAT'])) 
sum(pd.isnull(phillycrime['LONG'])) 

#Remove null values from dataframe 
data = pd.DataFrame(phillycrime, columns=['LAT','LONG'])
data = data.dropna()

#==============================================================================
# Optimize bandwidth
#==============================================================================
#Create 10,000 datapoint subset of the data to test for best bandwidth (otherwise runtime is very long)
bw_test_data = data.sample(n=10000,replace=False)
features = ['LAT', 'LONG']
bw_test_matrix = bw_test_data.as_matrix(columns = features)

##Test for best bandwidth
#params = {'bandwidth': np.logspace(-1., 1., 20)}
#grid = GridSearchCV(KernelDensity(), params)
#grid.fit(bw_test_matrix)
#print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth)) 
##On 2006 data, found that the best estimator was 0.1, since that is the lowest allowed by this set, I will now test a lower range, and not test again for other years since we'll assume the distribution  of the data will be similar enough
#params = {'bandwidth': np.logspace(-2., -1., 20)}
#grid = GridSearchCV(KernelDensity(), params)
#grid.fit(bw_test_matrix)
#print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth)) 
##Again, on 2006 data, found that the best estimator was 0.01, since that is the lowest allowed by this set, I will now test a lower range, and not test again for other years since we'll assume the distribution  of the data will be similar enough

params = {'bandwidth': np.logspace(-3., -2., 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(bw_test_matrix)
print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth)) 
best_bandwidths.append(grid.best_estimator_.bandwidth)
##best bandwidth first time on 2006 data:  0.00206913808111
##best bandwidth second time on 2006 data: 0.00183298071083
##best bandwidth third time on 2006 data: 0.00183298071083
#best bandwidth for 2007 first time: 0.00162377673919
#best bandwidth for 2007 second time: 0.00206913808111
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
# Get KDE samples for Contour plot
#==============================================================================

#Get KDE for the points in the plot
Z = np.exp(kde.score_samples(xy))
Z = Z.reshape(X.shape)

#==============================================================================
# Make Contour Plot and Save Image (for overlay)
#==============================================================================
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
plt.axis("off")
max_crime_density.append(Z.max())
levels = np.linspace(0, max(max_crime_density), 25) #Create 'hotness' levels for the contour plot
#maxZ 1st time I ran this code on 2006 data = 211.50162631233624
#maxZ 2nd time on 2006 data = 223.53529395143997  
#If we redo map with higher zoom value, we should set max level to the highest level of Z at Zoom level = 14 (showing the whole city), so that if we zoom in on a lower-crime area, we won't have the heat levels reset to relative levels

#Create contour plot
plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)

extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()) #gets extent of bbox for just the plot (without the frame/padding), so we can save just the plot extent (no frame / axis), allowing us to overlay the images later
plotname = 'philly%s_zoom%s_contour.png' % (year, zoom)
plt.savefig(plotname, bbox_inches=extent) #save image of plot

##==============================================================================
## Overlay Map tile and contour plots with Alpha Transparency on whole map
##==============================================================================
#import matplotlib.image as mpimg
#contourname = 'philly%s_zoom%s_contour.png' %(year, zoom)
#contour_img = mpimg.imread(contourname)
#mapname = 'philly_zoom%s_map.png' %(zoom) #We load the map for whatever zoom level the variable "zoom" is set to
#map_img = mpimg.imread(mapname)  
#
#fig = plt.figure(figsize=(12,12))
#ax = plt.subplot(111)
#extent = xmin, xmax, ymin, ymax
#contourlayer = plt.imshow(contour_img,interpolation="nearest",extent=extent)
#plt.hold(True)
#map_layer = plt.imshow(map_img, alpha=.4, interpolation='bilinear',extent=extent)
#plt.show()
#plotname = 'philly%s_zoom%s_alpha.png' % (year, zoom)
#plt.savefig(plotname) % (year) #save image of plot

#==============================================================================
# Playing around with setting transparency levels for colors of map
#==============================================================================
#Make white pixels in StamenToner tiles image transparent
from PIL import Image

mapname = 'philly_zoom%s_map.png' %(zoom) #We load the map for whatever zoom level the variable "zoom" is set to

map_img = Image.open(mapname)
map_img = map_img.convert("RGBA")
datas = map_img.getdata()

newData = []
for item in datas:
    if item[0] >200 and item[1] >200 and item[2] > 200:
        newData.append((item[0], item[1], item[2], 0))
    else:
        newData.append(item)
map_img.putdata(newData)

#masked_map_img = np.ma.masked_where(map_img != 0, map_img)
contourname = 'philly%s_zoom%s_contour.png' %(year, zoom)
contour_img = Image.open(contourname)
#contour_img = contour_img.convert("RGBA")
#extent = X.min(), X.max(), Y.min(), Y.max()
fig = plt.figure(figsize=(12,12))
ax = plt.subplot(111)
extent = xmin, xmax, ymin, ymax
contourlayer = plt.imshow(contour_img,interpolation="nearest",extent=extent)
plt.hold(True)
map_layer = plt.imshow(map_img, alpha=.9, interpolation='bilinear',extent=extent)
plt.show()
plotname = 'philly%s_zoom%s_masking.png' % (year, zoom)
plt.savefig(plotname) % (year) #save image of plot

            
