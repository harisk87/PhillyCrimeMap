"""
================================================
Kernel Density Estimate of Species Distributions
================================================
This shows an example of a neighbors-based query (in particular a kernel
density estimate) on geospatial species_data, using a Ball Tree built upon the
Haversine distance metric -- i.e. distances over points in latitude/longitude.
The species_dataset is provided by Phillips et. al. (2006).
If available, the example uses
`basemap <http://matplotlib.sourceforge.net/basemap/doc/html/>`_
to plot the coast lines and national boundaries of South America.

This example does not perform any learning over the species_data
(see :ref:`example_applications_plot_species_distribution_modeling.py` for
an example of classification based on the attributes in this species_dataset).  It
simply shows the kernel density estimate of observed species_data points in
geospatial coordinates.

The two species are:

 - `"Bradypus variegatus"
   <http://www.iucnredlist.org/apps/redlist/details/3038/0>`_ ,
   the Brown-throated Sloth.

 - `"Microryzomys minutus"
   <http://www.iucnredlist.org/apps/redlist/details/13408/0>`_ ,
   also known as the Forest Small Rice Rat, a rodent that lives in Peru,
   Colombia, Ecuador, Peru, and Venezuela.

References
----------

 * `"Maximum entropy modeling of species geographic distributions"
   <http://www.cs.princeton.edu/~schapire/papers/ecolmod.pdf>`_
   S. J. Phillips, R. P. Anderson, R. E. Schapire - Ecological Modelling,
   190:231-259, 2006.
"""
# Author: Jake Vanderplas <jakevdp@cs.washington.edu>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn.species_datasets import fetch_species_distributions
from sklearn.species_datasets.species_distributions import construct_grids
from sklearn.neighbors import KernelDensity

# if basemap is available, we'll use it.
# otherwise, we'll improvise later...
try:
    from mpl_toolkits.basemap import Basemap
    basemap = True
except ImportError:
    basemap = False

# Get matrices/arrays of species IDs and locations
species_data = fetch_species_distributions()
species_names = ['Bradypus Variegatus', 'Microryzomys Minutus']

speciesXtrain = np.vstack([species_data['train']['dd lat'],
                    species_data['train']['dd long']]).T
ytrain = np.array([d.decode('ascii').startswith('micro')
                  for d in species_data['train']['species']], dtype='int')
speciesXtrain *= np.pi / 180.  # Convert lat/long to radians

# Set up the species_data grid for the contour plot
xgrid, ygrid = construct_grids(species_data)
speciesX, speciesY = np.meshgrid(xgrid[::5], ygrid[::5][::-1])
land_reference = species_data.coverages[6][::5, ::5]
land_mask = (land_reference > -9999).ravel()

species_xy = np.vstack([speciesY.ravel(), speciesX.ravel()]).T
species_xy = species_xy[land_mask]
species_xy *= np.pi / 180.

# Plot map of South America with distributions of each species
fig = plt.figure()
fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05)

for i in range(2):
    plt.subplot(1, 2, i + 1)

    # construct a kernel density estimate of the distribution
    print(" - computing KDE in spherical coordinates")
    kde = KernelDensity(bandwidth=0.04, metric='haversine',
                        kernel='gaussian', algorithm='ball_tree')
    kde.fit(speciesXtrain[ytrain == i])

    # evaluate only on the land: -9999 indicates ocean
    speciesZ = -9999 + np.zeros(land_mask.shape[0])
    speciesZ[land_mask] = np.exp(kde.score_samples(species_xy))
    speciesZ = speciesZ.reshape(speciesX.shape)

    # plot contours of the density
    levels = np.linspace(0, speciesZ.max(), 25)
    plt.contourf(speciesX, speciesY, speciesZ, levels=levels, cmap=plt.cm.Reds)

    if basemap:
        print(" - plot coastlines using basemap")
        m = Basemap(projection='cyl', llcrnrlat=speciesY.min(),
                    urcrnrlat=speciesY.max(), llcrnrlon=speciesX.min(),
                    urcrnrlon=speciesX.max(), resolution='c')
        m.drawcoastlines()
        m.drawcountries()
    else:
        print(" - plot coastlines from coverage")
        plt.contour(speciesX, speciesY, land_reference,
                    levels=[-9999], colors="k",
                    linestyles="solid")
        plt.xticks([])
        plt.yticks([])

    plt.title(species_names[i])

plt.show()
