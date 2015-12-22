//Holds the VM projections
#ifndef FPGAVMPROJECTIONS_H
#define FPGAVMPROJECTIONS_H

#include "FPGATracklet.hh"
#include "FPGAMemoryBase.hh"

using namespace std;

class FPGAVMProjections:public FPGAMemoryBase{

public:

  FPGAVMProjections(string name, unsigned int iSector, 
		    double phimin, double phimax):
    FPGAMemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addTracklet(FPGATracklet* tracklet) {
    tracklets_.push_back(tracklet);
  }

  unsigned int nTracklets() const {return tracklets_.size();}

  FPGATracklet* getFPGATracklet(unsigned int i) const {return tracklets_[i];}

  void clean() {
    tracklets_.clear();
  }

private:

  double phimin_;
  double phimax_;
  std::vector<FPGATracklet*> tracklets_;

};

#endif
