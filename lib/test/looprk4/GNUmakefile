# -*- Mode: Makefile -*- 

### This makefile produces an executable for each name in the `ebase'
###  variable using the libraries named in the `LibNames' variable.
DIM=3

# included makefiles need an absolute path
CHOMBO_HOME := $(CHOMBO_LIBDIR)

# makefiles+=lib_test_MOLAMRTimeDependent

ebase := testRK4BEloop

LibNames := AMRTimeDependent AMRElliptic AMRTools BoxTools

base_dir = .
# src_dirs = $(base_dir)

include $(CHOMBO_HOME)/mk/Make.test

LIBFLAGS += $(LAPACKLIBS) 
# -lmpi

