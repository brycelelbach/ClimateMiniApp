# -*- Mode: Makefile -*- 

MPI=FALSE
DEBUG=TRUE
OPT=FALSE
DIM=3
CH_TIMER=TRUE

## Define the variables needed by Make.example

# trace the chain of included makefiles
makefiles += verticalSolvetest 

# the base name(s) of the application(s) in this directory
ebase = testVerticalSolve

# the location of Chombo lib dir
CHOMBO_HOME = $(CHOMBO_LIBDIR)

# names of Chombo libraries needed by this program, in order of search.
# MOLAMRTimeDependent has dups with HOMappedConservation, so put MOL last
LibNames := AMRElliptic AMRTools BoxTools BaseTools AMRTimeDependent

# relative paths to source code directories
base_dir = .

src_dirs = 
# \
# $(base_dir)/../util \
# $(base_dir)/../../src/advect \
# $(base_dir)/../../src/AMRTimeDependent \
# $(base_dir)/../../src/aniso3D \
# $(base_dir)/../../src/dycore3D \
# $(base_dir)/../../src/geometrysrc \
# $(base_dir)/../../src/loops \
# $(base_dir)/../../src/lwe \
# $(base_dir)/../../src/swe \
# $(base_dir)/../../src/verticalSolve \
# $(base_dir)/../../src/wave_equation

# input file for 'run' target
INPUT = inputs

# shared code for building example programs
include $(CHOMBO_HOME)/mk/Make.example

# additional libraries
LAPACKLIBS = -llapack
LIBFLAGS += $(LAPACKLIBS)

# application-specific variables
# This is needed for multi-block AMR.
XTRACXXFLAGS = -DDISABLE_TEMPORARY_FLUX_REGISTER_OPTIMIZATION

# application-specific targets
