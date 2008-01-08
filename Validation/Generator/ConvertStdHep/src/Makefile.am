## Process this file with automake to produce Makefile.in

if USE_GMAKE
  MAKE = gmake
endif

CLHEPdir = @CLHEPdir@

INCLUDES = -I$(top_builddir) \
	   -I$(top_srcdir) \
           -I$(top_srcdir)/stdhep/stdhep \
           -I$(top_srcdir)/mcfio/src \
	   -I$(top_srcdir)/HepMC \
	   -I$(CLHEPdir)/include

bin_PROGRAMS = convertStdHep

LDADD = $(top_builddir)/stdhep/src/libstdhepC.a \
        $(top_builddir)/mcfio/src/libFmcfio.a \
        $(top_builddir)/HepMC/src/libHepMC.a \
        $(CLHEPdir)/lib/libCLHEP.a

convertStdHep_SOURCES = convertHEPEVT.cc HEPEVT_CWrapper.cc convertStdHep.cc

