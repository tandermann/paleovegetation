#install.packages("remotes")
#remotes::install_github("macroecology/paleoMap")
# library(devtools)
# install_github("macroecology/mapast")
# install.packages("paleobioDB")
library(mapast)
library(paleobioDB)
library(sf)

#________________________________PALEOVEG DATA___________________________________________
# read the paleodata
datafile = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/vegetation_data/paleo_vegetation_north_america_albers.txt'
dataset = read.csv(datafile,sep='\t',stringsAsFactors = TRUE)
length_data = dim(dataset)[1]

# we need to bring the data into the right format for the function that only takes pbdb data
# 1. extract the columns we need for the paleocoord calculation
lat = dataset$Latitude
lng = dataset$Longitude
avg_age = dataset$mean_age
early_age = dataset$Age.Max
late_age = dataset$Age.Min
species = dataset$Openness
occurrence_no = dataset$index
matched_rank = rep('species', length_data)

# 2. load toy data to get the correct column names
toy_data  <-  base::data.frame(paleobioDB::pbdb_occurrences(base_name = "Canis", 
                                                            min_ma = 0, max_ma = 10, 
                                                            show = c("coords", "phylo"), 
                                                            vocab = "pbdb", limit = 100))

# 3. fill columns of toy data with paleoveg data
paleoveg_data = data.frame(matrix(ncol = length(colnames(toy_data)), nrow = length_data))
paleoveg_data$lng = lng
paleoveg_data$lat = lat
paleoveg_data$early_age = early_age
paleoveg_data$late_age = late_age
paleoveg_data$matched_rank = rep('species', length(lng))
paleoveg_data$matched_name = species
paleoveg_data$occurrence_no = occurrence_no
#paleoveg_data = data.frame(species,matched_rank,lng, lat, early_age,late_age,avg_age)


# 4. apply the formatdata function
df <- formatdata(data = paleoveg_data)

# we don't need to run the block below anymore, so it's commented out
## 5. run the paleocord calculation with midpoint of appearance time
model = "PALEOMAP"
occ_ma <- paleocoords(data = df, time = "average", model = model,stepsize=1)
#
## 6. remove unneccessary columns and format identical to before
start_col_index = length(colnames(occ_ma))-11
occ_data_selected_cols = occ_ma[,start_col_index:length(colnames(occ_ma))]
colnames(occ_data_selected_cols) = c('Longitude', 'Latitude', 'Age.Max', 'Age.Min', 'delete', 'Openness', 'index', 'delete', 'mean_age', 'age_bin','paleolon','paleolat')
occ_data_selected_cols = occ_data_selected_cols[,!(names(occ_data_selected_cols) == 'delete')]
col_order = c('index','Latitude','Longitude','Age.Max', 'Age.Min','Openness',  'mean_age', 'age_bin','paleolat','paleolon')
final_sorted_df = occ_data_selected_cols[,col_order]
final_sorted_df = final_sorted_df[order(final_sorted_df$index),]
write.table(final_sorted_df,'/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/vegetation_data/paleo_vegetation_data_north_america_with_paleocoords.txt',sep='\t',quote=FALSE,row.names=FALSE)

#_________________________________________________________________________________________







#________________________________CURRENT GRID COORDS___________________________________________
#library(mefa)

# 1. read the grid coordinates
'/Users/tobias/GitHub/paleovegetation/results/main_pipeline_out/area_-180_-52_25_80/coordinates_to_predict_past.txt'
grid_coords_file = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/vegetation_data/current_vegetation_north_america.txt'
grid_coords_file_content = read.table(grid_coords_file,header = TRUE)
# select coordinates
grid_coords = grid_coords_file_content[1:2]
length_grid_data = dim(grid_coords)[1]

# 2. extend the dataframe to the necessary dimensions
template_data_full = rep(df,5)
template_data_sized = template_data_full[1:length_grid_data,]

# 3. fill with our gridpoint data
time_column = rep(0, length_grid_data)
data_current = cbind(grid_coords,time_column,grid_coords)
df_current <- data.frame(data_current)
colnames(df_current) <- c('lng','lat','avg_age','paleolng','paleolat')
filename = paste0('/Users/tobias/GitHub/paleovegetation/data/processed/spatial_data/current_grid_with_paleocoords/grid_points_paleocoords_0MA.txt')
write.table(df_current,filename,sep='\t',quote=FALSE,row.names=FALSE)

# apply ecoregions shape file
# load ecoregions shape file
ecoregions_shp_file = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/ecoregions_shape/NA_CEC_Eco_Level1.shp'
ecoregions = st_read(ecoregions_shp_file)
# convert ecoregions shape into right projection
ecoregions_transformed = st_transform(ecoregions, projection('+proj=longlat +datum=WGS84 +no_defs'))
ecoregions_spatial = as(ecoregions_transformed, 'Spatial')
# aggregate by ecoregion
ecoregions_spatial_joined = aggregate(ecoregions_spatial, by='NA_L1NAME')
# define the gridcoords
grid_coords_spatial <- SpatialPoints(grid_coords)
crs(grid_coords_spatial) = '+proj=longlat +datum=WGS84 +no_defs'
ecoregion_info_all_points = over(grid_coords_spatial, ecoregions_spatial_joined)

# extract gridcoords for individual ecoregions
extract_ecoregion_spatial_points <- function(ecoregion_info_all_points,ecoregion_name){
  boolean_area = ecoregion_info_all_points == ecoregion_name
  is.na(boolean_area) = FALSE
  boolean_area[is.na(boolean_area)] = FALSE
  selected_coords = grid_coords[boolean_area,]
  selected_coords_spatial = SpatialPoints(selected_coords)
  return(selected_coords_spatial)
}

# identify cells that belong into each ecoregion
for (ecoregion_name in ecoregions_spatial_joined$NA_L1NAME){
  #ecoregion_name = 'NORTHERN FORESTS'
  ecoregion_name_file_string = gsub(' ','_',tolower(ecoregion_name))
  selected_points = extract_ecoregion_spatial_points(ecoregion_info_all_points,ecoregion_name)
  # write coords to file
  write.csv(selected_points,paste0('/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/ecoregion_cells_paleocoordinates/cells_by_ecoregion/',ecoregion_name_file_string,'.txt'),quote = FALSE,row.names = FALSE)
  # plot figure
  pdf(paste0('/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/ecoregion_cells_paleocoordinates/plots/',ecoregion_name_file_string,'.pdf'))
  plot(ecoregions_spatial_joined, col='grey', lwd=0.2)
  plot(ecoregions_spatial_joined[ecoregions_spatial_joined$NA_L1NAME==ecoregion_name,], col='light blue', lwd=0.2, add=T)
  plot(selected_points,add=T,col='red', pch=20,cex=0.3)
  dev.off()
}

# track these cells through time


'ecoregion_cells_paleocoordinates'



# now do the other time slices
timepoints = seq(1,30)
for (timepoint in timepoints){
  avg_age = rep(timepoint, length_grid_data)
  template_data_sized$avg_age = avg_age
  template_data_sized$lng = grid_coords$x
  template_data_sized$lat = grid_coords$y
  template_data_sized = template_data_sized
  # run the paleocoord calculation
  model = "PALEOMAP"
  gridpoint_paleocoords_df = paleocoords(data = template_data_sized, time = "average", model = model,stepsize=1)
  # write to file
  target_cols = c('lng','lat','avg_age','paleolng','paleolat')
  gridpoint_paleocoords_df = gridpoint_paleocoords_df[,target_cols]
  filename = paste0('/Users/tobias/GitHub/paleovegetation/data/processed/spatial_data/current_grid_with_paleocoords/grid_points_paleocoords_',timepoint,'MA.txt')
  write.table(gridpoint_paleocoords_df,filename,sep='\t',quote=FALSE,row.names=FALSE)
}













#__________________EXAMPLE: TRANSFORM COORDS INTO PALEO COORDS________________________
# toy data
data  <-  base::data.frame(paleobioDB::pbdb_occurrences(base_name = "Canis", 
                                                        min_ma = 0, max_ma = 10, 
                                                        show = c("coords", "phylo"), 
                                                        vocab = "pbdb", limit = 100))
# format for paleocoords function
df <- formatdata(data = data)
# explanation of models: https://github.com/GPlates/gplates_web_service_doc/wiki/Reconstruction-Models
# paleomap more specifically: https://www.earthbyte.org/paleomap-paleoatlas-for-gplates/
model = "PALEOMAP"
#reconstruct paleocoordinates with midpoint of appearance time
occ_ma <- paleocoords(data = df, time = "automatic", model = model)
#reconstruct paleocoordinates with specific time
occ_matime <- paleocoords(data = df, time = "timevector", timevector = c(65), model = model) #---> doesn't work
#create a plot with fossils on the paleogeographical map
mapast(model = model, data = df_auto)
#_____________________________________________________________________________





