#! /bin/bash

XSIZE=1440
YSIZE=960
SIZE=${XSIZE}x${YSIZE}
STEPS=`ls -1 ./numeric.*.dat | wc -l`

# every INCth step is rendered
INC=1

FPS_FACTOR=25

for movie in numeric analytic error
do
    echo "RENDERING `echo ${movie} | tr '[:upper:]' '[:lower:]'`"

    mkdir -p scratch-${movie}
    cd scratch-${movie}

    rm step.*.jpeg > /dev/null 2>&1

    seq 0 $INC $(($STEPS-1)) | parallel --eta "gnuplot -e 'NAME=\"$movie\"' -e STEPS=$STEPS -e XSIZE=$XSIZE -e YSIZE=$YSIZE -e 'STEP={}' ../render_frame.gpi"

    ffmpeg -y -pattern_type glob -r $(($FPS_FACTOR/$INC)) -i './step.*.jpeg' -s $SIZE -b:v 4096k -r 24 ${movie}.mpeg
    mv -f ${movie}.mpeg ..

    cd ..
done

