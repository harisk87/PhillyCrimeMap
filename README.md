Modified 06/20/2016 Hari
————

# Notes

-Removed all MATLAB code and figures

-Created a new file PhillyCrimeMap.ipynb to aggregate all findings into a Jupyter notebook

-Added new kde plot files

-Data picked into file crime_data

# Accomplishments

-Learned to use OpenDataPhilly's API to request for most up-to-date crime data

-Used Seaborn's kdeplot function to reduce the task of the plotting the heat maps to a single line of code. Seaborn plots also look cleaner

-Used Z-tests and t-tests to test the following hypotheses

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-There are on average more aggravated crimes per day in the Spring and Summer than Fall and Winter (very low p-value using Z-test)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-There are on average more thefts per day in the Spring and Summer than Fall and Winter (very low p-value using Z-test)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-There are fewer aggravated crimes per day when it's cold (i.e. in winter) than during the rest of the year (very low p-value using t-test)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-There are fewer thefts per day when it's cold (i.e. in winter) than during the rest of the year (very low p-value using t-test)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-There are on average more aggravated crimes per night than per day (very low p-value using t-test)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-There are on average more thefts per night than per day (Cannot reject null hypothesis. Opposite might be true, but need to take into account time theft is reported)

# Future Work

-Use box and violin plots to visualize the data (cleaner than using histograms)
-Study each type of crime in more detail to see if there are patterns
-Think of more classifiers for crime such as area of the crime
-Use the heat maps to study how probable it is for a given crime to have been reported in a given region
-Find regions of high specific crimes (prositution, drug rings etc) and see if a pattern exists to where these types of crimes happen


Created: 12/3/2015  Heather

This project is an ongoing collaborative effort between Heather Simpson and Hari Sivakumar

We aim to create a visualization of how the severity and localization of crime in the city of Philadelphia has changed over the past 10 years. 

We will use a Gaussian kernel density estimate to create a heat contour map of the crime 'hotspots'.

Heather has created the .R and .py files here, Hari has created a version of the kde and contour plot in Matlab.

Data Set is freely available from: 
https://www.opendataphilly.org/dataset/philadelphia-police-part-one-crime-incidents


# Philly_Crime_Map.py 
     	- Plots circles representing crime incidents onto map 

	- uses StamenToner OpenStreetMap tiles

	- calls StamenTonerTilesAccess.py (adapted from OpenStreetMap tile code provided here: http://stackoverflow.com/questions/28476117/easy-openstreetmap-tile-displaying-for-python)

	- Scikit-learn kde model fit onto the data

	- contour plot created with matplotlib 

# Philly_Crime_Map.R 

	- Plots circles representing crime incidents onto map 

	- Using Leaflet to make it interactive
————
#Ideas/Notes/Plans: 

Map Size/View: 
	- probably we should create several static views of the map 

	- 1 view for each major neighborhood + 1 overview view?

Map Slider (to go through diff. years)

	in R:  maybe try using Shiny to add the slider function on top of Leaflet plot

	in Python : MatPlot has slider function

