library(raster)
library(rworldmap)
library(sp)
library(maps)
library(mapdata)

world_map <- getMap(resolution = "less islands")
world_map = aggregate(world_map,dissolve=T)
e <- extent(-180, 180, -55, 80) # world
world_map <- crop(world_map, e)
plot(world_map)
map.axes()
#plot(spatial_points,add=TRUE,cex=1,col='red',pch = 19)
#plot(SpatialPoints(cbind(175.25, -36.75)),add=TRUE,cex=1,col='red',pch = 19)


precipitation_folder = '/Users/tobias/GitHub/paleovegetation_mammals/data/raw/climate_data/precipitation_0_540Ma/interpolated/'
precipitation_files = dir(precipitation_folder, pattern = "*interpolated.txt", full.names = TRUE, ignore.case = TRUE)
for (precipitation_t in precipitation_files){
  precipitation_t_split = strsplit(precipitation_t, '/')[[1]]
  basename = precipitation_t_split[length(precipitation_t_split)]
  step = gsub("Ma_precipitation_interpolated.txt","",basename)
  time_slice = strtoi(step,base = 10)

  #read precipitation file
  precipitation = read.csv(precipitation_t,sep=',',header = FALSE)
  prec_values = array(do.call(c, unlist(list(precipitation), recursive=FALSE)))
  #prec_values = prec_values[1:32400]
  
  # create raster and assign precipitation values to it
  prec_raster = raster(nrow=360,ncol=180,xmn=-90, xmx=90, ymn=-180, ymx=180,res=1)
  values(prec_raster) = c(prec_values)
  prec_raster = t(prec_raster)

  # convert from resolution 1 to resolution 0.5 to match the other data
  new_prec_raster = disaggregate(prec_raster, fact=2)

  # select only land cells
  #prec_land = mask(new_prec_raster, world_map)
  # let's use all of the grid, to make sure we have everything covered
  prec_land = new_prec_raster

  # extract cell-centers and export as coordinates
  spts <- rasterToPoints(prec_land, spatial = TRUE)
  spts = as(spts,'SpatialPoints')
  data_points = as.data.frame(spts)
  values = extract(prec_land,data_points)
  # assign the precipitation value to the coordinates
  data_points$prec=values
  # write to file
  filename_table = paste0('/Users/tobias/GitHub/paleovegetation_mammals/data/processed/climate_data/precipitation/global_precipitation_',time_slice,'ma.txt')
  write.table(data_points,filename_table,sep='\t',quote = FALSE,row.names = FALSE)  
  
  # plot the raster and save as file
  cuts = c(0.2,0.4,0.6,0.8,1,2,3,4,5,6,30)
  colors = colorRampPalette(c("yellow","darkgreen"))
  outfilename = paste0('/Users/tobias/GitHub/paleovegetation_mammals/fig/precipitation_maps/global_precipitation_',time_slice,'ma.png')
  title = paste0(time_slice,' Ma')
  
  png(outfilename,width = 2000, height = 1200)
  plot(prec_land, xlim=c(e[1],e[2]), ylim=c(e[3],e[4]), breaks=cuts, col=colors(length(cuts)) )
  plot(world_map,add=T,lwd=2)
  title(main=title, cex.main=10,line = -7)
  dev.off()
  
}





temperature_folder = '/Users/tobias/GitHub/paleovegetation_mammals/data/raw/climate_data/temperature_0_540Ma/interpolated/'
temp_files = dir(temperature_folder, pattern = "*interpolated.txt", full.names = TRUE, ignore.case = TRUE)
for (temperature_t in temp_files){
  temperature_t_split = strsplit(temperature_t, '/')[[1]]
  basename = temperature_t_split[length(temperature_t_split)]
  step = gsub("Ma_temperature_interpolated.txt","",basename)
  time_slice = strtoi(step,base = 10)
  
  #read temperature file
  temperature = read.csv(temperature_t,sep=',',header = FALSE)
  temp_values = array(do.call(c, unlist(list(temperature), recursive=FALSE)))
  #prec_values = prec_values[1:32400]
  
  # create raster and assign precipitation values to it
  temp_raster = raster(nrow=360,ncol=180,xmn=-90, xmx=90, ymn=-180, ymx=180)
  values(temp_raster) = c(temp_values)
  temp_raster = t(temp_raster)
  

  
  # convert from resolution 1 to resolution 0.5 to match the other data
  new_temp_raster = disaggregate(temp_raster, fact=2)
  
  # select only land cells
  #temp_land = mask(new_temp_raster, world_map)
  # let's use all of the grid, to make sure we have everything covered
  temp_land = new_temp_raster  
  
  # extract cell-centers and export as coordinates
  spts <- rasterToPoints(temp_land, spatial = TRUE)
  spts = as(spts,'SpatialPoints')
  data_points = as.data.frame(spts)
  values = extract(temp_land,data_points)
  # assign the precipitation value to the coordinates
  data_points$temp=values
  # write to file
  filename_table = paste0('/Users/tobias/GitHub/paleovegetation_mammals/data/processed/climate_data/temperature/global_temperature_',time_slice,'ma.txt')
  write.table(data_points,filename_table,sep='\t',quote = FALSE,row.names = FALSE)  
  
  # plot the raster and save as file
  cuts = c(-10,-5,0,5,10,15,20,25,30,35,40,100)
  colors = colorRampPalette(c("yellow","darkgreen"))
  
  outfilename = paste0('/Users/tobias/GitHub/paleovegetation_mammals/fig/temperature_maps/global_temperature_',time_slice,'ma.png')
  title = paste0(time_slice,' Ma')
  
  png(outfilename,width = 2000, height = 1200)
  plot(temp_land, xlim=c(e[1],e[2]), ylim=c(e[3],e[4]), breaks=cuts, col=colors(length(cuts)) )
  plot(world_map,add=T,lwd=2)
  title(main=title, cex.main=10,line = -7)
  dev.off()
}
