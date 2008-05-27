#include <iostream>
#include <fstream>
#include <sstream>

#include "SimDataFormats/GeneratorProducts/interface/AlpWgtFileInfoProduct.h"

using namespace edm;
using namespace std;

// copy constructor
AlpWgtFileInfoProduct::AlpWgtFileInfoProduct(AlpWgtFileInfoProduct const& other) {
    seed1_ = other.seed1();
    seed2_ = other.seed2();
    wgt1_ = other.wgt1();
    wgt2_ = other.wgt2();
}

void AlpWgtFileInfoProduct::AddEvent(const char* buffer) {
  char seed1_c[512];  
  char seed2_c[512];
  char one_c[511];
  char wgt1_c[512];
  char wgt2_c[512];
  
  istringstream is(buffer);
  is >> seed1_c >> seed2_c >> one_c >> wgt1_c >> wgt2_c;
  seed1_.push_back(atoi(seed1_c));
  seed2_.push_back(atoi(seed2_c));
  wgt1_.push_back(atof(wgt1_c));
  wgt2_.push_back(atof(wgt2_c));
}
