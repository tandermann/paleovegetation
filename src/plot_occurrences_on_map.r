indir='/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/fossil_data/occurrences_by_taxon/current'
file_list = list.files(indir)
outpath = '/Users/tobias/GitHub/feature_gen_paleoveg/data/raw/fossil_data/occurrences_by_taxon/plots'
library(raster)
library(rworldmap)
library(sp)
library(sf)
#install.packages('rworldxtra')

plot(newmap)
newmap <- getMap(resolution = "high")

pdf("/Users/tobiasandermann/Desktop/map.pdf")
plot(newmap, xlim=c(-180,-52), ylim=c(25,80),asp=1)
dev.off()


for(i in file_list){
  database = read.csv(paste0(indir,'/',i),sep = '\t')
  taxonname = sub('_current_occs.txt','',i)

  lat = as.character(database$lat)
  lon = as.character(database$lon)

  outfile = paste0(outpath,'/',taxonname,'_gbif_occs.pdf')
  pdf(outfile)
  plot(newmap, xlim=c(-180,-52), ylim=c(25,80),asp=1)
  points(lon, lat, col = "red", cex = .2)
  dev.off()
}

for(i in file_list){
  database = read.csv(paste0(indir,'/',i),sep = '\t')
  taxonname = sub('_current_occs.txt','',i)
  
  lat = as.character(database$lat)
  lon = as.character(database$lon)
  
  outfile = paste0(outpath,'/',taxonname,'_gbif_occs.pdf')
  pdf(outfile)
  plot(newmap, xlim=c(-180,-52), ylim=c(25,80),asp=1)
  points(lon, lat, col = "red", cex = .2)
  dev.off()
}


# load map in albers projection
world_map <- getMap(resolution = "low")
e <- extent(-180, -52, 25, 80)
north_america <- crop(world_map, e)
crs(north_america)
albers_proj = "+init=EPSG:5070"
north_america_albers = spTransform(north_america, projection(albers_proj))
plot(north_america_albers)

# load the taxon occurrences (past and present)
all_taxon_occs_file = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/data_for_distance_extraction/final_occs_present_past_binned_ages_albers.txt'
all_taxon_occs = read.csv(all_taxon_occs_file,sep='\t')
occs = SpatialPoints(cbind(all_taxon_occs$x_albers,all_taxon_occs$y_albers))
crs(occs) = albers_proj

# load paleovegetation labels (past and present)
paleoveg_file = '/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/raw/vegetation_data/selected_paleo_vegetation_north_america_albers.txt'
paleoveg_data = read.csv(paleoveg_file,sep='\t')
paleoveg_pts = SpatialPoints(cbind(paleoveg_data$x_albers,paleoveg_data$y_albers))
crs(paleoveg_pts) = albers_proj

# bin by geological stage
age_bins = table(all_taxon_occs$rounded_ages)
age_bins_names = as.numeric(names(age_bins))

# plot 1 map per stage
pdf('/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/data_for_distance_extraction/plots/all.pdf')
par(mfrow=c(5,3),mar = c(1, 1, 1, 1))
for (i in age_bins_names){
  selected_taxon_occs = all_taxon_occs[all_taxon_occs$rounded_ages==i,]
  occs = SpatialPoints(cbind(selected_taxon_occs$x_albers,selected_taxon_occs$y_albers))
  crs(occs) = albers_proj
  selected_paleoveg_data = paleoveg_data[paleoveg_data$rounded_ages==i,]
  closed_labels = selected_paleoveg_data[selected_paleoveg_data$label==0,]
  open_labels = selected_paleoveg_data[selected_paleoveg_data$label==1,]
  #paleoveg_pts_raw = cbind(selected_paleoveg_data$x_albers,selected_paleoveg_data$y_albers)
  closed_paleoveg_pts_raw = cbind(closed_labels$x_albers,closed_labels$y_albers)
  open_paleoveg_pts_raw = cbind(open_labels$x_albers,open_labels$y_albers)
  filename = paste0("/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/data_for_distance_extraction/plots/map_bin_",i,".pdf")
#  pdf(filename)
  plot(north_america_albers,col="grey",border="grey",main=paste0(i,' Ma'))
  plot(occs,add=T,pch=20, col = "red", cex = .3)
  if (length(closed_paleoveg_pts_raw)>0){
    closed_paleoveg_pts = SpatialPoints(closed_paleoveg_pts_raw)
    plot(closed_paleoveg_pts,add=T,pch=21, col = "darkgreen", cex = 1.)
  }
  if (length(open_paleoveg_pts_raw)>0){
    open_paleoveg_pts = SpatialPoints(open_paleoveg_pts_raw)
    plot(open_paleoveg_pts,add=T,pch=21, col = "goldenrod", cex = 1.)
  }
#  dev.off()
}
dev.off()


# plot 1 map per stage
pdf('/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/data_for_distance_extraction/plots/all_datatype.pdf')
par(mfrow=c(5,3),mar = c(1, 1, 1, 1))
for (i in age_bins_names){
  selected_taxon_occs = all_taxon_occs[all_taxon_occs$rounded_ages==i,]
  occs = SpatialPoints(cbind(selected_taxon_occs$x_albers,selected_taxon_occs$y_albers))
  crs(occs) = albers_proj
  selected_paleoveg_data = paleoveg_data[paleoveg_data$rounded_ages==i,]
  closed_labels = selected_paleoveg_data[selected_paleoveg_data$label==0,]
  open_labels = selected_paleoveg_data[selected_paleoveg_data$label==1,]
  #paleoveg_pts_raw = cbind(selected_paleoveg_data$x_albers,selected_paleoveg_data$y_albers)
  closed_paleoveg_pts_raw = cbind(closed_labels$x_albers,closed_labels$y_albers)
  open_paleoveg_pts_raw = cbind(open_labels$x_albers,open_labels$y_albers)
  filename = paste0("/Users/tobiasandermann/GitHub/feature_gen_paleoveg/data/data_for_distance_extraction/plots/map_bin_",i,".pdf")
  #  pdf(filename)
  plot(north_america_albers,col="grey",border="grey",main=paste0(i,' Ma'))
  plot(occs,add=T,pch=20, col = "red", cex = .3)
  if (length(closed_paleoveg_pts_raw)>0){
    closed_paleoveg_pts = SpatialPoints(closed_paleoveg_pts_raw)
    plot(closed_paleoveg_pts,add=T,pch=21, col = "darkgreen", cex = 1.)
  }
  if (length(open_paleoveg_pts_raw)>0){
    open_paleoveg_pts = SpatialPoints(open_paleoveg_pts_raw)
    plot(open_paleoveg_pts,add=T,pch=21, col = "goldenrod", cex = 1.)
  }
  #  dev.off()
}
dev.off()






pdf("/Users/tobiasandermann/Desktop/map.pdf")
plot(north_america_albers,col="grey",border="black")
plot(occs,add=T,pch=20, col = "red", cex = .1)
plot(paleoveg_pts,add=T,pch=20, col = "darkgreen", cex = .5)
dev.off()





