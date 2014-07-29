#! /bin/bash

export HPX_LOCATION="$HOME/install/hpx/gcc-4.8-debug"
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HPX_LOCATION/lib/pkgconfig

mkdir -p BUILD_LOGS

LOG=BUILD_LOGS/build_`date "+%Y.%m.%d_%H.%M.%S"`.log

ln -fs $LOG build_latest.log

make $@ 2>&1 | tee $LOG 

