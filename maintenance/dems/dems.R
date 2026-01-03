# 1. Download from Datamart
# 2. Rename
# 3. Use wgrib2 to set_grib_type complex3 (https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/set_grib_type.html)

require(terra)
require(sf)

hrw <- rast("G:/My Drive/snowmodels/ops/nwp_archives/dems/HRDPS_WEST_DEM.grib2")
hro <- rast("G:/My Drive/snowmodels/ops/nwp_archives/dems/HRDPS_OLD_DEM.grib2")
hr <- rast("G:/My Drive/snowmodels/ops/nwp_archives/dems/HRDPS_DEM.grib2")
rdps <- rast("G:/My Drive/snowmodels/ops/nwp_archives/dems/RDPS_DEM.grib2")
# NEW RDPS: 20250613T06Z_MSC_RDPS_GeopotentialHeight_Sfc_RLatLon0.09_PT000H.grib2
gdps <- rast("G:/My Drive/snowmodels/ops/nwp_archives/dems/GDPS_DEM.grib2")
fx <- st_read("https://raw.githubusercontent.com/avalanche-canada/forecast-polygons/refs/heads/main/editting_tools/reference_regions.geojson")

par(mfrow = c(2, 2))
plot(hrw, main = "HRDPS WEST", axes = FALSE)
lines(st_transform(fx, crs(hrw)))
plot(hr, main = "HRDPS", axes = FALSE)
lines(st_transform(fx, crs(hr)))
plot(rdps, main = "RDPS", axes = FALSE)
lines(st_transform(fx, crs(rdps)))
plot(gdps, main = "GDPS", axes = FALSE)
lines(st_transform(fx, crs(gdps)))

dim(hrw)
dim(hro)
dim(hr)
dim(rdps)
dim(gdps)
st_crs(crs(hrw))$proj
st_crs(crs(hro))$proj
st_crs(crs(hr))$proj
st_crs(crs(rdps))$proj
st_crs(crs(gdps))$proj
