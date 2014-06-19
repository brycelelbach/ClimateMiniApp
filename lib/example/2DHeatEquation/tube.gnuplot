nx=100.0
ny=10.0

set size ratio ny/nx

set terminal png enhanced size 1440,720 font ",16"

set pm3d map

set xtics (0.0, nx*(2.0/4.0), nx)
set ytics (0.0, ny*(2.0/4.0), ny)

unset key
set tics nomirror out

set colorbox horizontal
set colorbox user origin 0.1,0.15 size 0.8,0.1

set samples nx
set isosamples nx,ny

A=0.25
a=500.0/(nx*nx)
c=20.0/(ny*ny)

x0=(nx-1.0)*(1.0/2.0)
y0=(ny-1.0)*(1.0/2.0)

set xlabel "Horizontal"
set ylabel "Vertical"

set xrange [0:nx-1]
set yrange [0:ny-1]

print "A = ", A
print "a = ", a
print "c = ", c
print "x0 = ", x0
print "y0 = ", y0

set output 'tube.png'
splot A*exp(a*(x-x0)-c*(abs(y-y0)))
