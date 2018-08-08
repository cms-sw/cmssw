GCC=g++ -g `root-config --cflags --glibs` $< -o $@


all : singlePi.exe RelValMacro.so

singlePi.exe : SinglePi.C
	$(GCC)

RelValMacro.so : RelValMacro.C
	$(eval tpypath := $(shell python get.py '$(CMSSW_FWLITE_INCLUDE_PATH)'))
	g++ -fPIC -shared RelValMacro.C -o RelValMacro.so -I $(tpypath)/include/ -I /cvmfs/cms.cern.ch/slc6_amd64_gcc491/external/python/2.7.6-eccfad/include/python2.7 -L /cvmfs/cms.cern.ch/slc6_amd64_gcc491/external/python/2.7.6-eccfad/lib -lpython2.7 -L $(tpypath)/lib -lboost_python -g `root-config --cflags --glibs`
