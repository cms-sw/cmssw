# Makefile for compiling standalone MET validation programs
# by Mike Schmitt (The University of Florida) July 23 2007

CC=g++

CFLAGS=-Wall -c -g -O
LFLAGS=-Wall -lCore -lCint -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -lpthread -lm -ldl -rdynamic

LIBS=-L$(ROOTSYS)/lib
INCLUDES=-I$(ROOTSYS)/include -Iinclude/

SOURCES=$(wildcard src/*plotCompare.cpp)
PACKAGES=$(SOURCES:src/%.cpp=%)
BINARIES=$(PACKAGES:%=bin/%)

PCU=obj/PlotCompareUtility.o
HD=obj/HistoData.o
TDR=obj/TDRStyle.o
OBJECTS=$(PACKAGES:%=obj/%.o) $(PCU) $(TDR) $(HD)

all: $(BINARIES) $(OBJECTS)

clean:
	rm -f $(BINARIES) $(OBJECTS)

obj/%.o: src/%.cpp include/%.h
#	@echo $(@:obj/%.o=%) updating: $?
	$(CC) $(CFLAGS) $(INCLUDES) $(@:obj/%.o=src/%.cpp) -o $@

obj/%.o: src/%.cpp
#	@echo $(@:obj/%.o=%) updating: $?
	$(CC) $(CFLAGS) $(INCLUDES) $(@:obj/%.o=src/%.cpp) -o $@

bin/%: obj/%.o $(PCU) $(TDR) $(HD)
#	@echo $(@:bin/%=%) linking
	$(CC) $(LFLAGS) $(LIBS) $(@:bin/%=obj/%.o) $(PCU) $(TDR) $(HD) -o $@

%: bin/%
	@echo $@ has been successfully built

