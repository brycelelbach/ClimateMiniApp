# -*- Mode: Makefile -*- 

### This makefile produces an executable for each name in the `ebase'
###  variable using the libraries named in the `LibNames' variable.

# included makefiles need an absolute path
CHOMBO_HOME := $(shell pwd)/../..

makefiles+=lib_example_ClimateMiniApp

ebase = ClimateMiniApp 

src_dirs = $(base_dir)/../../src/AMRTimeDependent

# note that BaseTools library should be included by default, even 
# if we don't specify it here
ifeq ($(RUNTIME),HPX)
	LibNames = BoxTools AMRTools AsyncTools
else
	LibNames = BoxTools AMRTools
endif

include $(CHOMBO_HOME)/mk/Make.example
