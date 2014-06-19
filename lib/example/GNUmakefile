# -*- Mode: Makefile -*- 

export makefiles+=lib_example

## This variable specifies where we are.  Lower-level make files will need an
## absolute path so we use 'pwd' rather than "."
export CHOMBO_HOME := $(shell pwd)/..

##
## Default values of configuration variables.
##
include $(CHOMBO_HOME)/mk/Make.defs.defaults

##
## These variables define all the subdirectories that can be built.
## These library names are used by the makefiles in the application directories.
##[NOTE: if you change the names, change the dependencies below.]

example_targets := $(shell find example -mindepth 1 -maxdepth 1 -type d)

##
## These are the actions (targets for the recursive make) supported by this
## makefile.  It assumes the makefiles in the subdirectories support them too.
##   all        builds all the executables, including the libraries they depend on
##   clean      deletes files for this configuration
##   realclean  deletes files for all configurations
##   vars       prints values of various make variables
##
_all_actions = all clean realclean

.PHONY: $(_all_actions) $(example_targets) usage vars

##
## Default target
##
usage:
	@echo "This makefile supports targets:"
	@echo "     $(_all_actions) $(example_targets) usage vars"

##
## For each action, set the action variable to tell the recursive
## makefile what to do.  There should be a separate dependency for
## every value in $(_all_actions).
##[NOTE: the `NODEPENDS=TRUE' variable disables automatic
##       dependency analysis in the recursive makefile, which is only
##       needed for actions that compile or link code.]
##
_action = all
all       : _action = all
clean     : _action = clean     NODEPENDS=TRUE
realclean : _action = realclean NODEPENDS=TRUE

# this dependency ensures the action is done for every directory.
$(_all_actions) : $(example_targets)

# this rule performs the action
$(example_targets):
	$(MAKE) --no-print-directory --directory $@ $(_action)

# this rule prints out various 'make' variables to the screen for debugging
vars:
	@$(MAKE) --no-print-directory -f $(CHOMBO_HOME)/mk/Make.printVariables $@

