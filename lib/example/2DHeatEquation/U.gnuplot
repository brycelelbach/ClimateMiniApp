OUTPUT=(sprintf('step.%04i.jpeg', STEP))
INPUT=(sprintf('../U.%04i.dat', STEP))
INIT=(sprintf('../U.0000.dat', STEP))

stats INIT nooutput
X_MIN=STATS_min_x
X_MAX=STATS_max_x
Y_MIN=STATS_min_y
Y_MAX=STATS_max_y

nx=X_MAX-X_MIN
ny=Y_MAX-Y_MIN

min(x,y) = (x<y)?x:y
max(x,y) = (x>y)?x:y

#stats 'soln_initial.dat' using 3 nooutput name "INIT"
#stats 'soln_final.dat' using 3 nooutput name "FINI"
#max_T=max(ceil(INIT_max), ceil(FINI_max))
#min_T=min(floor(INIT_min), floor(FINI_min))
max_T=-100000000
min_T=100000000
do for [i=0:STEPS-1] {
    I_DATA=(sprintf('../U.%04i.dat', i))
    stats I_DATA using 3 nooutput 
    max_T=max(ceil(STATS_max+0.001), max_T)
    min_T=min(floor(STATS_min), min_T)
}

#print "min_T = ", min_T
#print "max_T = ", max_T

#print "ny = ", ny
#print "nx = ", nx

set terminal jpeg size XSIZE,YSIZE

set size ratio ny/nx

#set pm3d map

set view 60,60
#show view

set xtics (X_MIN, X_MIN+nx*(2.0/4.0), X_MAX)
set ytics (Y_MIN, Y_MIN+ny*(2.0/4.0), Y_MAX)

unset key
set tics nomirror out

#set colorbox horizontal
#set colorbox user origin 0.1,0.1 size 0.8,0.05

set samples X_MAX-X_MIN
set isosamples X_MAX-X_MIN,Y_MAX-Y_MIN

set xlabel "Horizontal"
set ylabel "Vertical"

set xrange[X_MIN:X_MAX]
set yrange[Y_MIN:Y_MAX]

set cbrange[min_T:max_T]
set zrange [min_T:max_T]

set output OUTPUT
splot INPUT using 1:2:3 with pm3d 
