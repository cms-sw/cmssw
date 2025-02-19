
CC = g++

all: lateralAllCalib

lateralAllCalib: lateralAllCalib.cpp
	$(CC) lateralAllCalib.cpp -o lateralAllCalib `root-config --libs` -lMinuit -I`root-config --incdir` -I.

clean:
	rm -f lateralAllCalib

