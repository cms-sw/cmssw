## Process this file with automake to produce Makefile.in

if USE_GMAKE
  MAKE = gmake
endif

includedir = $(prefix)/include
COPY_P = @COPY_P@

pkginclude_HEADERS = \
                   convertHEPEVT.h \
		   HEPEVT_CWrapper.h \
                   defs.h

# Identify generated file(s) to be removed when 'make clean' is requested:
CLEANFILES = defs.h
