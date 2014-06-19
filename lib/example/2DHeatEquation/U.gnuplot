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
stats INIT using 3 nooutput name "INIT"
max_T=ceil(INIT_max)
min_T=floor(INIT_min)

#print "ny = ", ny
#print "nx = ", nx

set terminal jpeg size 1440,720

set size ratio ny/nx

set pm3d map

set xtics (X_MIN, X_MIN+nx*(2.0/4.0), X_MAX)
set ytics (Y_MIN, Y_MIN+ny*(2.0/4.0), Y_MAX)

unset key
set tics nomirror out

set colorbox horizontal
set colorbox user origin 0.1,0.15 size 0.8,0.1

set samples X_MAX-X_MIN
set isosamples X_MAX-X_MIN,Y_MAX-Y_MIN

set xlabel "Horizontal"
set ylabel "Vertical"

set xrange[X_MIN:X_MAX]
set yrange[Y_MIN:Y_MAX]

set cbrange[min_T:max_T]

set output OUTPUT
splot INPUT using 1:2:3 
