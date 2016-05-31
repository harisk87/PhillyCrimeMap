# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:31:35 2016

@author: heathersimpson
"""
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.text as txt
import matplotlib.animation as animation

zoom = 12
years = ['2006','2007','2008', '2009','2010','2011', '2012','2013', '2014']

#==============================================================================
# Plot Philly Crime Map with Animation
#==============================================================================
mywriter = animation.FFMpegWriter()
fig = plt.figure(figsize=(13,13))
ax = fig.add_subplot(111)
plotims = []
for year in years:
    plotname = 'philly%s_zoom%s_overlay.png' % (year, zoom)
    overlayim = Image.open(plotname)
    im = plt.imshow(overlayim)
    bbox_props = dict(boxstyle="round",fc="lightgray",ec="dimgray",lw=2)
    t = ax.annotate(s = year, xy=(75,170), xycoords='data',xytext=(0.1,0.1),textcoords='offset points', bbox=bbox_props, family = 'sans-serif', fontsize = 70, color='darkred', weight = 'bold')
    plotims.append([im,t])

ani = animation.ArtistAnimation(fig, plotims, interval=950, repeat_delay=10000, blit=False)

#ani.save('phillycrime_zoom12_animated.mp4', writer=mywriter)
#plt.show()

plt.close()