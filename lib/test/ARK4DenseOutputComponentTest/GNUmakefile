# -*- Mode: Makefile -*- 

### This makefile produces an executable for each name in the `ebase'
###  variable using the libraries named in the `LibNames' variable.

# included makefiles need an absolute path
CHOMBO_HOME := $(shell pwd)/../..

makefiles+=lib_test_ARK4DenseOutputComponentTest

ebase := testARK4DenseOutputComponentTest

LibNames := AMRElliptic AMRTools BoxTools

#base_dir = .
#src_dirs = $(base_dir)/../../src/AMRTimeDependent

include $(CHOMBO_HOME)/mk/Make.test

