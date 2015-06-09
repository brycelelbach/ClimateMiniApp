# -*- Mode: Makefile -*- 

### This makefile produces an executable for each name in the `ebase'
###  variable using the libraries named in the `LibNames' variable.

# included makefiles need an absolute path
# CHOMBO_HOME := $(shell pwd)/../..
CHOMBO_HOME := $(CHOMBO_LIBDIR)

makefiles+=lib_test_ARK4

ebase := testARK4VerticalSolve

LibNames := AMRElliptic AMRTools BoxTools 

base_dir = .
#src_dirs = $(base_dir)/../../src/MiscUtil

include $(CHOMBO_HOME)/mk/Make.test

# additional libraries
LAPACKLIBS = -llapack -lblas
LIBFLAGS += $(LAPACKLIBS)


