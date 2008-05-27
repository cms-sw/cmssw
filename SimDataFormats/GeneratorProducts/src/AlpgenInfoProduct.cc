#include <sstream>

#include "SimDataFormats/GeneratorProducts/interface/AlpgenInfoProduct.h"

using namespace edm; 
using namespace std;

AlpgenInfoProduct::AlpgenInfoProduct (int n) : nEv_(0) {}

// copy constructor
AlpgenInfoProduct::AlpgenInfoProduct(AlpgenInfoProduct const& other) :
  nEv_(other.nEv()), nTot_(other.nTot()), subproc_(other.subproc()), q_(other.Q()) {
  lundIn_ = other.lundIn();
  colorIn_ = other.colorIn();
  colorBarIn_ = other.colorBarIn();
  pzIn_ = other.pzIn();
  lundOut_ = other.lundOut();
  colorOut_ = other.colorOut();
  colorBarOut_ = other.colorBarOut();
  pxOut_ = other.pxOut();
  pyOut_ = other.pyOut();
  pzOut_ = other.pzOut();
  massOut_ = other.massOut();
}


void AlpgenInfoProduct::EventInfo(const char* buffer){

  char   nEv_char[80];
  char   subproc_char[80];
  char   nTot_char[80];
  char   idk2_char[80];
  char   q_char[80];
  
  istringstream is(buffer);
  is >> nEv_char >>  subproc_char  >> nTot_char >> idk2_char >> q_char;
  nEv_  = atoi(nEv_char);
  subproc_ = atoi(subproc_char);
  nTot_ = atoi(nTot_char);
  q_  = atof(q_char);
}

void AlpgenInfoProduct::InPartonInfo(const char* buffer){

  char lundIn_char[80];
  char colorIn_char[80];
  char colorBarIn_char[80];
  char pzIn_char[80];

  istringstream is(buffer);
  is >> lundIn_char >>  colorIn_char >>  colorBarIn_char >>  pzIn_char;
  lundIn_.push_back(atoi(lundIn_char));  
  colorIn_.push_back(atoi(colorIn_char));  
  colorBarIn_.push_back(atoi(colorBarIn_char));  
  pzIn_.push_back(atof(pzIn_char));
}


void AlpgenInfoProduct::OutPartonInfo(const char* buffer){
 
  char lundOut_char[80];
  char colorOut_char[80];
  char colorBarOut_char[80];
  char pxOut_char[80];
  char pyOut_char[80];
  char pzOut_char[80];
  char massOut_char[80];

  istringstream is(buffer);
  is >> 
    lundOut_char >> colorOut_char >> colorBarOut_char >> 
    pxOut_char >> pyOut_char >> pzOut_char >> massOut_char;

  lundOut_.push_back(atoi(lundOut_char)); 
  colorOut_.push_back(atoi(colorOut_char)); 
  colorBarOut_.push_back(atoi(colorBarOut_char)); 
  pxOut_.push_back(atof(pxOut_char)); 
  pyOut_.push_back(atof(pyOut_char)); 
  pzOut_.push_back(atof(pzOut_char)); 
  massOut_.push_back(atof(massOut_char));
}
