// This class holds a list of stubs that are in a given layer and DCT region
#ifndef FPGASTUBDISK_H
#define FPGASTUBDISK_H

#include "L1TStub.hh"
#include "FPGAStub.hh"
#include "FPGAMemoryBase.hh"

using namespace std;

class FPGAStubDisk:public FPGAMemoryBase{

public:

  FPGAStubDisk(string name, unsigned int iSector, 
		double phimin, double phimax):
    FPGAMemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addStub(std::pair<FPGAStub*,L1TStub*> stub) {
    stubs_.push_back(stub);
  }

  unsigned int nStubs() const {return stubs_.size();}

  FPGAStub* getFPGAStub(unsigned int i) const {return stubs_[i].first;}
  L1TStub* getL1TStub(unsigned int i) const {return stubs_[i].second;}
  std::pair<FPGAStub*,L1TStub*> getStub(unsigned int i) const {return stubs_[i];}

  void clean() {
    stubs_.clear();
  }

  void writeStubs(bool first) {

    std::string fname="DiskStubs_";
    fname+=getName();
    fname+="_";
    ostringstream oss;
    oss << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=oss.str();
    fname+=".dat";
    if (first) {
      bx_=0;
      event_=1;
      out_.open(fname.c_str());
    }
    else
      out_.open(fname.c_str(),std::ofstream::app);

    out_ << "BX = "<<(bitset<3>)bx_ << " Event : " << event_ << endl;

    for (unsigned int j=0;j<stubs_.size();j++){
      string stub=stubs_[j].first->str();
      if (j<16) out_ <<"0";
      out_ << hex << j << dec ;
      out_ <<" "<<stub << endl;
    }
    out_.close();

    bx_++;
    event_++;
    if (bx_>7) bx_=0;

  }

private:

  double phimin_;
  double phimax_;
  std::vector<std::pair<FPGAStub*,L1TStub*> > stubs_;

};

#endif
