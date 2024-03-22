
PYTHON_INCLUDE = /cvmfs/cms.cern.ch/el9_amd64_gcc12/external/python3/3.9.14-8e02587b42992e07ed46b00eca9dfc3a/include/python3.9/
PYTHON_LIB     = /cvmfs/cms.cern.ch/el9_amd64_gcc12/external/python3/3.9.14-8e02587b42992e07ed46b00eca9dfc3a/lib
BOOST_INC      = /cvmfs/cms.cern.ch/el9_amd64_gcc12/external/boost/1.80.0-e4b2127ad40e46379cea2f9087beeda8/include
BOOST_LIB      = /cvmfs/cms.cern.ch/el9_amd64_gcc12/external/boost/1.80.0-e4b2127ad40e46379cea2f9087beeda8/lib

TARGET = RelValMacro_ext

$(TARGET).so: $(TARGET).o
	g++ -shared -Wl,--export-dynamic $(TARGET).o -L$(BOOST_LIB) -lboost_python39 -L${PYTHON_LIB} -lpython3.9 -o $(TARGET).so -g `root-config --cflags --glibs`
$(TARGET).o: $(TARGET).C
	g++ -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $(TARGET).C  -g `root-config --cflags --glibs`

