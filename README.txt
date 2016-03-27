README

Created: 12/3/2015  Heather
————
This project is an ongoing collaborative effort between Heather Simpson and Hari Sivakumar

We aim to create a visualization of how the severity and localization of crime in the city of Philadelphia has changed over the past 10 years. 

We will use a Gaussian kernel density estimate to create a heat contour map of the crime 'hotspots'.

Heather has created the .R and .py files here, Hari has created a version of the kde and contour plot in Matlab.
— 

Data Set is freely available from: 
https://www.opendataphilly.org/dataset/philadelphia-police-part-one-crime-incidents

—
————
Philly_Crime_Map.py 
     	- Plots circles representing crime incidents onto map 
	- uses StamenToner OpenStreetMap tiles
	- calls StamenTonerTilesAccess.py (adapted from OpenStreetMap tile code provided here: http://stackoverflow.com/questions/28476117/easy-openstreetmap-tile-displaying-for-python)
	- Scikit-learn kde model fit onto the data
	- contour plot created with matplotlib 

Philly_Crime_Map.R 
	- Plots circles representing crime incidents onto map 
	- Using Leaflet to make it interactive
————
Ideas/Notes/Plans: 

Map Size/View: 
	- probably we should create several static views of the map 
	- 1 view for each major neighborhood + 1 overview view?

Map Slider (to go through diff. years)
	in R:  maybe try using Shiny to add the slider function on top of Leaflet plot
	in Python : MatPlot has slider function

Ideas for Analyzing/Plotting Data: 
	- use a kernel density analysis / plop a Gaussian on the points to create a smoothed set of values?
	- try Voronoi polygons for the heatmap shapes

	- define large-scale clusters with K-means, then use DBscan within those areas
	- use DBscan with different settings to build up clusters of different sizes
	- analyze density (just # of crimes/area) for small areas and define hotspots based on 	
	that


