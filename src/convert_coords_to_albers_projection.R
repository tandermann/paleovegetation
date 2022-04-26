library(sf)
library(raster)
#library(rgdal)
setwd('/Users/tobiasandermann/GitHub/feature_gen_paleoveg/')


convert_lonlat_to_albers <- function(lonlat_coords,plot_maps=FALSE){
  spts = SpatialPoints(lonlat_coords,proj4string = CRS('+proj=longlat +datum=WGS84 +no_defs'))
  albers_proj = "+init=EPSG:5070"
  spts_albers = spTransform(spts, projection(albers_proj))
  albers_occs = data.frame(spts_albers)
  if (plot_maps==TRUE){
    cropping_window = t(data.frame(list(c(-180,25),c(-52,80))))
    row.names(cropping_window) = c(1,2)
    spts_wind = SpatialPoints(cropping_window,proj4string = CRS('+proj=longlat +datum=WGS84 +no_defs'))
    spts_wind_albers = spTransform(spts_wind, projection(albers_proj))
    wind_coords = data.frame(spts_wind_albers)
    
    png('data/data_for_distance_extraction/lonlat_coords.png')
    plot(spts)
    axis(1, at=NULL, labels=TRUE)
    axis(2, at=NULL, labels=TRUE)
    plot(spts_wind, cex=2, pch=20, col='red',add=T)
    dev.off()
    
    png('data/data_for_distance_extraction/albers_coords.png')
    plot(spts_albers,xlim = c(min(wind_coords$coords.x1), max(albers_occs$coords.x1)))
    axis(1, at=NULL, labels=TRUE)
    axis(2, at=NULL, labels=TRUE)
    plot(spts_wind_albers, cex=2, pch=20, col='red',add=T)
    dev.off()
    print(wind_coords)
  }
  return(albers_occs)
}


occ_coords_file = 'data/data_for_distance_extraction/final_occs_present_past.txt'
occ_data = read.csv(occ_coords_file,sep = '\t')
lonlat_occs = cbind(occ_data$lon,occ_data$lat)
albers_occs = convert_lonlat_to_albers(lonlat_occs,plot_maps=TRUE)

occ_data_albers = occ_data
occ_data_albers$x_albers = albers_occs$coords.x1
occ_data_albers$y_albers = albers_occs$coords.x2

occ_data_albers$lon <- round(as.numeric(occ_data_albers$lon),2)
occ_data_albers$lat <- round(as.numeric(occ_data_albers$lat),2)
occ_data_albers$mean_age <- round(as.numeric(occ_data_albers$mean_age),2)
occ_data_albers$x_albers <- round(as.numeric(occ_data_albers$x_albers),0)
occ_data_albers$y_albers <- round(as.numeric(occ_data_albers$y_albers),0)


write.table(occ_data_albers,
            file = 'data/data_for_distance_extraction/final_occs_present_past_albers.txt',
            sep = '\t',
            row.names = FALSE,
            col.names = TRUE,
            quote = FALSE)


# transform coords for paleoveg file
paleo_veg_file = 'data/raw/vegetation_data/paleo_vegetation_north_america.txt'
paleo_veg_data = read.csv(paleo_veg_file,sep = '\t')
# current_coords
lonlat_paleo_veg = cbind(paleo_veg_data$Longitude,paleo_veg_data$Latitude)
albers_paleo_veg = convert_lonlat_to_albers(lonlat_paleo_veg)
paleo_veg_data$x_albers = round(as.numeric(albers_paleo_veg$coords.x1,0))
paleo_veg_data$y_albers = round(as.numeric(albers_paleo_veg$coords.x2,0))
# paleocoords
lonlat_paleo_veg = cbind(paleo_veg_data$paleolon,paleo_veg_data$paleolat)
albers_paleo_veg = convert_lonlat_to_albers(lonlat_paleo_veg)
paleo_veg_data$x_paleo_albers = round(as.numeric(albers_paleo_veg$coords.x1,0))
paleo_veg_data$y_paleo_albers = round(as.numeric(albers_paleo_veg$coords.x2,0))
# write to file
write.table(paleo_veg_data,
            file = 'data/raw/vegetation_data/paleo_vegetation_north_america_albers.txt',
            sep = '\t',
            row.names = FALSE,
            col.names = TRUE,
            quote = FALSE)


# transform coords for current veg file
current_veg_file = 'data/raw/vegetation_data/current_vegetation_north_america.txt'
current_veg_data = read.csv(current_veg_file,sep = '\t')
# current_coords
lonlat_current_veg = cbind(current_veg_data$x,current_veg_data$y)
albers_current_veg = convert_lonlat_to_albers(lonlat_current_veg)
current_veg_data$x_albers = round(as.numeric(albers_current_veg$coords.x1,0))
current_veg_data$y_albers = round(as.numeric(albers_current_veg$coords.x2,0))
# write to file
write.table(current_veg_data,
            file = 'data/raw/vegetation_data/current_vegetation_north_america_albers.txt',
            sep = '\t',
            row.names = FALSE,
            col.names = TRUE,
            quote = FALSE)


# transform coords for past spatial grids
target_folder = 'data/raw/current_grid_with_paleocoords/lonlat/'
txtfiles <- paste('grid_points_paleocoords_', 0:30, 'MA.txt', sep='')
paleo_coord_files <- paste(target_folder, txtfiles, sep='')
for (paleocoordfile in paleo_coord_files){
  coords_data = read.csv(paleocoordfile,sep = '\t')
  lonlat_coords = cbind(coords_data$lng,coords_data$lat)
  albers_coords = convert_lonlat_to_albers(lonlat_coords)
  coords_data$x_present_albers = round(as.numeric(albers_coords$coords.x1,0))
  coords_data$y_present_albers = round(as.numeric(albers_coords$coords.x2,0))
  # write to file
  write.table(coords_data,
              file = gsub('grid_points_paleocoords','../grid_points_paleocoords_albers',paleocoordfile),
              sep = '\t',
              row.names = FALSE,
              col.names = TRUE,
              quote = FALSE)
}


