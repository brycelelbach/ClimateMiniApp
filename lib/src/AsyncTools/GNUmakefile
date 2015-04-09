#      _______              __
#     / ___/ /  ___  __ _  / /  ___
#    / /__/ _ \/ _ \/  V \/ _ \/ _ \
#    \___/_//_/\___/_/_/_/_.__/\___/
#
#    Please refer to Copyright.txt, in Chombo's root directory.

# -*- Mode: Makefile -*- 

### This makefile produces a library for the name in the 'lbase' variable
### depending on the libraries named in the 'LibNames' variable

# included makefiles need an absolute path
CHOMBO_HOME := $(shell pwd)/../..

lbase := asynctools

LibNames := BoxTools AMRTools

# put this first so 'Cfile' isn't the default target
include $(CHOMBO_HOME)/mk/Make.lib

.PHONY: Cfile

# >.< /s/ wash
Cfile:
	$(RM) CFile.Cpp CFile.h
	grep -h -v 'include' *.cpp > CFile.Cpp
	grep -h -v 'include' *.h > CFile.h
	cp -p CFile.Cpp ~/AMR-CCA/CFile.cpp
	cp -p CFile.h   ~/AMR-CCA/CFile.h
