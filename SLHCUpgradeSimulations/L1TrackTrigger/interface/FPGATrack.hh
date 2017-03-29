#ifndef FPGATRACK_HH
#define FPGATRACK_HH

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
#include <map>

using namespace std;

class FPGATrack{

public:

  FPGATrack(int irinv, int iphi0, int it, int iz0, 
	    std::map<int, int> stubID,
	    std::vector<L1TStub*> l1stub){

    irinv_=irinv;
    iphi0_=iphi0;
    iz0_=iz0;
    it_=it;
    stubID_=stubID;
    l1stub_=l1stub;
    duplicate_=false;
    isector_=28;

  }


  ~FPGATrack() {

  }
  
  void setDuplicate(bool flag) { duplicate_=flag; }
  void setSector(int nsec) { isector_=nsec; }

  int irinv() const { return irinv_; }
  int iphi0() const { return iphi0_; }
  int iz0()   const { return iz0_; }
  int it()    const { return it_; }
  std::map<int, int> stubID() const { return stubID_; }
  std::vector<L1TStub*> stubs() const { return l1stub_; }
  int duplicate() const { return duplicate_; }
  int isector() const { return isector_; }

  double z0() const {return iz0_*kzpars; } //in cm
  double rinv() const {return irinv_*krinvpars; } //in cm-1
  double pt(double bfield) const {return 0.003*bfield/rinv(); }
  double eta() const {return asinh(it_*ktpars);}
  double d0() const {return 0.0;} //Fix when fit for 5 pars
  double phi0() const {return iphi0_*kphi0pars+isector_*two_pi/NSector-two_pi/(6.0*NSector);}
  double chisq() const {return -1.0;}

private:
  
  int irinv_;
  int iphi0_;
  int iz0_;
  int it_;
  std::map<int, int> stubID_;
  std::vector<L1TStub*> l1stub_;
  bool duplicate_;
  int isector_;

};

#endif



