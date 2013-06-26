# dont overwrite this here, otherwise clang can't insert his "custom compiler"
#CXX = g++#/build/gcc/gcc-4.7.0-bin/bin/g++

all: class_checker.o mutable_member.o const_cast.o const_cast_away.o global_static.o static_local.o global_static_edm.o

mutable_member.o: mutable_member.cpp
	$(CXX) mutable_member.cpp

const_cast.o: const_cast.cpp
	$(CXX) const_cast.cpp

const_cast_away.o: const_cast_away.cpp
	$(CXX) const_cast_away.cpp
	
global_static.o: global_static.cpp
	$(CXX) global_static.cpp

global_static_edm.o: global_static_edm.cpp
	$(CXX) global_static_edm.cpp

static_local.o: static_local.cpp
	$(CXX) static_local.cpp

class_checker.o: class_checker.cpp
	$(CXX) -std=c++0x class_checker.cpp	
