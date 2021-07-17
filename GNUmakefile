AMREX_HOME = ../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_MPI   = FALSE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_DPCPP = FALSE

BL_NO_FORT = TRUE

TINY_PROFILE = FALSE

programs ?=

ifeq ($(strip $(programs)),)
  programs += branch/main
  programs += compute-bound/main
  programs += daxpy/main
  programs += max/main
  programs += reduce/main
  programs += scan/main
  programs += jacobi/main
  programs += jacobi_sync/main
  programs += jacobi_elixir/main
  programs += jacobi_memorypool/main
  programs += aos_simple/main
  programs += aos_shared/main
  programs += gsrb/main
  programs += parser/main
endif

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

multiple_executables = $(addsuffix .$(machineSuffix).ex, $(programs))
default: $(multiple_executables)

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

clean::
	$(RM) $(multiple_executables)
