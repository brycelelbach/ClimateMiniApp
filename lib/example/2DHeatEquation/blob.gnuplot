set terminal png enhanced size 960,960 font ",16"

set size square

set pm3d map

unset key
set tics nomirror out
unset ztics

nx=100
ny=100

set samples nx
set isosamples nx,ny

A=1.5
a=100.0/(nx*nx)
b=0.0
c=100.0/(ny*ny)

x0=(nx-1.0)/2.0
y0=(ny-1.0)/2.0

set xlabel "Horizontal"
set ylabel "Vertical"

set xrange [0:nx-1]
set yrange [0:ny-1]

print "A = ", A
print "a = ", a
print "b = ", b
print "c = ", c
print "x0 = ", x0
print "y0 = ", y0

set output 'blob.png'
splot A*exp(-(a*(x-x0)*(x-x0)+2*b*(x-x0)*(y-y0)+c*(y-y0)*(y-y0)))
