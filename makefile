XERCES_DIR = /cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/xerces-c/3.1.3
CLHEP_DIR = /cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/clhep/2.3.4.2-mlhled

OPT_PAR_DIR = /afs/cern.ch/user/j/jkaspar/software/optics_generator/CMSSW_9_1_0_pre2

INC = -I$(OPT_PAR_DIR)/interface
LIB = -L$(OPT_PAR_DIR)/lib -lFit -L$(XERCES_DIR)/lib -lxerces-c -L$(CLHEP_DIR)/lib -lCLHEP

all: test_reconstruction get_optical_functions

clean:
	rm -f test_reconstruction
	rm -f get_optical_functions

test_reconstruction : test_reconstruction.cc track_lite.h beam_conditions.h proton_reconstruction.h
	g++ -O3 -g -Wall -Wextra -Wno-attributes -Werror --std=c++11\
		`root-config --libs` `root-config --cflags`  \
		$(INC) $(LIB) \
		-Wl,-rpath=${XERCES_DIR}/lib \
		-Wl,-rpath=${CLHEP_DIR}/lib \
		-Wl,-rpath=${OPT_PAR_DIR}/lib \
			test_reconstruction.cc -o test_reconstruction

get_optical_functions : get_optical_functions.cc beam_conditions.h
	g++ -O3 -g -Wall -Wextra -Wno-attributes -Werror --std=c++11\
		`root-config --libs` `root-config --cflags` \
		$(INC) $(LIB) \
		-Wl,-rpath=${XERCES_DIR}/lib \
		-Wl,-rpath=${CLHEP_DIR}/lib \
		-Wl,-rpath=${OPT_PAR_DIR}/lib \
			get_optical_functions.cc -o get_optical_functions
