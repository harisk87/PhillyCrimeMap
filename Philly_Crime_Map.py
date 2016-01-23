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
zoom = 11

#lat_deg, lon_deg, delta_lat,  delta_long, zoom = lat_deg-delta_lat/2, lon_deg-delta_long/2, delta_lat,  delta_long, 11
#lat_deg, lon_deg, delta_lat,  delta_long, zoom = 45.720-0.04/2, 4.210-0.08/2, 0.04,  0.08, 14
a, bbox = STT.getImageCluster(lat_deg, lon_deg, delta_lat,  delta_long, zoom)

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111)
m = Basemap(
    llcrnrlon=bbox[0], llcrnrlat=bbox[1],
    urcrnrlon=bbox[2], urcrnrlat=bbox[3],
    projection='merc', ax=ax
)
# list of points to display (long, lat)
ls_points = [m(x,y) for x,y in lonlat]
m.imshow(a, interpolation='lanczos', origin='upper')
ax.scatter([point[0] for point in ls_points],
           [point[1] for point in ls_points],
           alpha = 0.9)
plt.show()

   