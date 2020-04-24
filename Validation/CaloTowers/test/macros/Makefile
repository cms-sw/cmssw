GCC=g++ -g `root-config --cflags --glibs` $< -o $@

all : singlePi.exe RelValMacro.exe

singlePi.exe : SinglePi.C
	$(GCC)

RelValMacro.exe : RelValMacro.C
	$(GCC)
