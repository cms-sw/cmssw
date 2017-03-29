#ifndef FPGAINVERSETABLE_H
#define FPGAINVERSETABLE_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class FPGAInverseTable{

public:

  FPGAInverseTable() {
   
  }

  ~FPGAInverseTable() {

  }


  void init(int nbits,
	    int offset
	    ) {

    nbits_=nbits;
    entries_=(1<<nbits);
    
    for(int i=0;i<entries_;i++) {
      int idrrel=i;
      if (i>((1<<(nbits-1))-1)) {
	idrrel=i-(1<<nbits);
      }
      int idr=offset+idrrel;
      table_.push_back(round_int((1<<idrinvbits)/(1.0*idr)));
    }


  }
	    

  void write(std::string fname) {

    ofstream out(fname.c_str());

    for (int i=0;i<entries_;i++){
      //cout << "i "<<i<<endl;
      out <<table_[i]<<endl;
    }
    out.close();
  
  }


  unsigned int lookup(int drrel) const {
    assert(drrel>=0);
    assert(drrel<(1<<nbits_));
    return table_[drrel];
  }

  int round_int( double r ) {
    return (r > 0.0) ? (r + 0.5) : (r - 0.5); 
  }


private:

  int nbits_;
  int entries_;
  vector<unsigned int> table_;
  

};



#endif



