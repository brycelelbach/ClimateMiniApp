#! /bin/bash

mkdir -p scratch
cd scratch

rm step.*.jpeg > /dev/null 2>&1

SIZE=1440x720
STEPS=5

# every INCth step is rendered
INC=1

FPS_FACTOR=25

seq 1 $INC $(($STEPS-1)) | parallel --eta 'gnuplot -e "STEP={}" ../U.gnuplot'

ffmpeg -y -pattern_type glob -r $(($FPS_FACTOR/$INC)) -i './step.*.jpeg' -s $SIZE -b:v 4096k -r 24 U.mpeg
mv -f U.mpeg ..

cd ..

