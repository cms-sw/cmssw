
PYTHON_INCLUDE = /cvmfs/cms.cern.ch/el9_amd64_gcc11/external/python3/3.9.14-e432d7f95f9e22c05899c3205f44ed54/include/python3.9
PYTHON_LIB     = /cvmfs/cms.cern.ch/el9_amd64_gcc11/external/python3/3.9.14-e432d7f95f9e22c05899c3205f44ed54/lib
BOOST_INC      = /cvmfs/cms.cern.ch/el9_amd64_gcc11/external/boost/1.80.0-d4d128fb3f8cdd37c4b1575d7f90e3dc/include
BOOST_LIB      = /cvmfs/cms.cern.ch/el9_amd64_gcc11/external/boost/1.80.0-d4d128fb3f8cdd37c4b1575d7f90e3dc/lib

TARGET = RelValMacro_ext

$(TARGET).so: $(TARGET).o
	g++ -shared -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python39 -L${PYTHON_LIB} -lpython3.9 -o $(TARGET).so -g `root-config --cflags --glibs`
$(TARGET).o: $(TARGET).C
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $(TARGET).C  -g `root-config --cflags --glibs`

