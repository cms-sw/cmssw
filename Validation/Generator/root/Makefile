# make file to compile analysis macros
ROOTLIBS      = $(shell root-config --libs)
ROOTGLIBS     = $(shell root-config --glibs)

LIBS          = $(ROOTLIBS)
GLIBS         = $(ROOTGLIBS) -L/usr/X11R6/lib -lXext -lX11


CXX           = gcc

CXXFLAGS      = -fPIC -Wall -Wno-deprecated -m32
LDFLAGS       = -fPIC -Wl -m32

LD_LIBRARY_PATH:= $(ROOTSYS)/lib/:$(LD_LIBRARY_PATH)


INCLUDEDIR    = -I$(ROOTSYS)/include -I./

OBJS          = MyMakePlots.o MyMakePlotsDict.o

install: $(OBJS)
	$(CXX) -shared -o libMyMakePlots.so  $(LDFLAGS) $(OBJS) $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDEDIR)
%.o: %.C
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDEDIR)

%Dict.cc: %.h %LinkDef.h
	@echo "Generating dictionary $@"
	@rootcint -f $@  -c  $^

clean:
	@rm -f $(OBJS) core *~ *.o *.so *.eps *.root

