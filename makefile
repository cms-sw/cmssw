all: test_reconstruction

test_reconstruction : test_reconstruction.cc track_lite.h beam_conditions.h proton_reconstruction.h
	g++ -O3 -g -Wall -Wextra -Wno-attributes -Werror --std=c++11\
		`root-config --libs` `root-config --cflags` -lMinuit2 \
		-I$(CMSSW_BASE)/src \
		-L$(CMSSW_BASE)/lib/slc6_amd64_gcc481/ \
		-lSimG4CMSTotemRPProtTranspPar \
			test_reconstruction.cc -o test_reconstruction
