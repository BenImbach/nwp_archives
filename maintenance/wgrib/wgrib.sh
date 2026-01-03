#!/bin/bash
# https://bitbucket.org/sfu-arp/sarpnwptools/src/master/wgrib2/installWgrib2.sh

## Old method works best
## https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/compile_questions.html
wget https://ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz
tar -xzvf wgrib2.tgz
sudo rm -r wgrib2.tgz && mv grib2 wgrib2 && cd wgrib2
export CC=gcc && export FC=gfortran && export COMP_SYS=gnu_linux
sed -i "s/USE_NETCDF3=1/USE_NETCDF3=0/g" makefile
make

## Different versions
WGRIB="/home/shorton/wgrib2/wgrib2/wgrib2" # Simon's Acer WSL
WGRIB="/shared/nwp_archives/wgrib2/wgrib2/wgrib2" # AWS worker
WGRIB="/home/shorton/sarpnwptools/wgrib2/wgrib2/wgrib2" # SFU Fir


# ## Aux progs
# [shorton@cedar2 wgrib2]$ ls aux_progs/
# README    gmerge.make  grib_split.c         rd_grib2_msg.c  smallest_4.make   smallest_grib2.make  wgrib2mv
# gmerge    gmerge_13.c  grib_split.make      smallest_4      smallest_grib2    uint8.c
# gmerge.c  grb2.h       gribtab2gribtable.c  smallest_4.c    smallest_grib2.c  wgrib2ms
# https://www.ftp.cpc.ncep.noaa.gov/wd51we/wgrib2_aux_progs/
# https://www.ftp.cpc.ncep.noaa.gov/wd51we/wgrib2_aux_progs/gmerge/


## Auxillary programs 

# # Download auxillary programs to run wgrib2 in parallel (wgrib2ms and wgrib2mv)
# wget http://www.ftp.cpc.ncep.noaa.gov/wd51we/wgrib2_aux_progs/wgrib2ms -P ${WGRIB2_DIR}/aux_progs
# wget http://www.ftp.cpc.ncep.noaa.gov/wd51we/wgrib2_aux_progs/wgrib2mv -P ${WGRIB2_DIR}/aux_progs
# chmod +x ${WGRIB2_DIR}/aux_progs/wgrib2ms
# chmod +x ${WGRIB2_DIR}/aux_progs/wgrib2mv
# # Change lines in script that point to wgrib2 and gmerge executables
# sed -i "67s/.*/wgrib2='\$WGRIB2_DIR\/wgrib2\/wgrib2'/" ${WGRIB2_DIR}/aux_progs/wgrib2ms
# sed -i "65s/.*/gmerge='\$WGRIB2_DIR\/aux_progs\/gmerge'/" ${WGRIB2_DIR}/aux_progs/wgrib2ms
# sed -i "51s/.*/wgrib2='\$WGRIB2_DIR\/wgrib2\/wgrib2'/" ${WGRIB2_DIR}/aux_progs/wgrib2mv
# sed -i "49s/.*/gmerge='\$WGRIB2_DIR\/aux_progs\/gmerge'/" ${WGRIB2_DIR}/aux_progs/wgrib2mv
