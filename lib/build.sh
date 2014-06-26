#! /bin/bash

mkdir -p BUILD_LOGS

LOG=BUILD_LOGS/build_`date "+%Y.%m.%d_%H.%M.%S"`.log

ln -fs $LOG build_latest.log

make $@ 2>&1 | tee $LOG 

