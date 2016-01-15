# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 19:57:59 2015

@author: heathersimpson
"""
#Using Shapely: import shapely
#import fiona
#import pprint
#
#with fiona.open("/Users/heathersimpson/Documents/Data_Science/Philly_Crime/neighborhoods_Azavea/neighborhoods_Azavea.shp") as src:
#    pprint.pprint(src[1])
#
#    philly_shape= shapely.geometry.shape(f['geometry'])
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import StamenTonerTilesAccess as STT


philly06 = pd.read_csv('GIS_POLICE.INCIDENTS_2006.csv')

# Find Max and Min Latitude value
ymax = philly06['POINT_Y'].max()  #Northmost point : 40.137445
ymin = philly06['POINT_Y'].min()  #Southmost point : 39.875032
# Find Max and Min Longitude value
xmax = philly06['POINT_X'].max()  #Eastmost point : -74.957504
xmin = philly06['POINT_X'].min()#Westmost point : -75.27426
lonlat = zip(philly06.POINT_X.values, philly06.POINT_Y.values)
lat_deg = ymax
lon_deg = xmin
delta_lat = ymax-ymin
delta_long = xmax-xmin
zoom = 11
a, bbox = STT.getImageCluster(lat_deg, lon_deg, delta_lat,  delta_long, zoom) 

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111)
m = Basemap(
    llcrnrlon=bbox[0], llcrnrlat=bbox[1],
    urcrnrlon=bbox[2], urcrnrlat=bbox[3],
    projection='merc', ax=ax
)
ls_points = [m(x,y) for x,y in lonlat]  # Need to fix so it can take lists of points as input, right now it's fixed
m.imshow(a, interpolation='nearest', origin='upper')
ax.scatter([point[0] for point in ls_points],
           [point[1] for point in ls_points],
           alpha = 0.9)
plt.show()


   