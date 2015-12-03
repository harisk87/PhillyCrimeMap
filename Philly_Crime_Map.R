library(leaflet)

crime06<-read.table(file="/Users/heathersimpson/Documents/Data_Science/Philly_Crime/GIS_POLICE.INCIDENTS_2006.csv", sep="," , quote = "\"", header=T, fill = T)
str(crime06)
#crime06<-(crime06[complete.cases(crime06),])
names(crime06)[names(crime06)=="POINT_X"]<-"Long"
names(crime06)[names(crime06)=="POINT_Y"]<-"Lat"
crime06<-crime06[is.na(crime06$Lat)==FALSE,] 
homfire<-crime06[crime06$UCR_GENERAL<=500,] #Get crimes that aren't theft
theft<-crime06[crime06$UCR_GENERAL>500,]
m<-leaflet(crime06) %>% setView(lng= -75.1667, lat= 39.95, zoom=13)
m %>% addProviderTiles("Stamen.Toner") %>% addCircles(lng= homfire$Long, lat = homfire$Lat, weight = 1, radius = 3, opacity=.9)
L.heatlayer


