cd /mnt/g/My\ Drive/snowmodels/ops/nwp_archives
GF=/mnt/c/Users/horto/Data/nwp_archives/out/HRDPS_20230223.grib2
NF=/mnt/c/Users/horto/Data/nwp_archives/out/HRDPS_20230223
WGRIB="/home/shorton/wgrib2/build/wgrib2/wgrib2"
NCTABLE=wgrib/nc_table.ini

# http://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/netcdf.html
time $WGRIB $GF -nc_table $NCTABLE -netcdf ${NF}_3.nc
## 953 MB grib becomes 2520 MB netcdf in 22 min (2.6x bigger)
time $WGRIB $GF -nc_table $NCTABLE -nc4 -netcdf ${NF}.nc
## 953 MB grib becomes 1032 MB netcdf in 2 min (1.1x bigger)

# Check results
ncdump -h ${NF}_3.nc
ncdump -h ${NF}.nc



GF=/mnt/c/Users/horto/Data/nwp_archives/out/HRDPS_20121202.grib2
NF=/mnt/c/Users/horto/Data/nwp_archives/out/HRDPS_20121202
time $WGRIB $GF -nc_table $NCTABLE -nc4 -netcdf ${NF}.nc


