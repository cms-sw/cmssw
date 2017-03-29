//Holds the candidate matches
#ifndef FPGAFULLMATCH_H
#define FPGAFULLMATCH_H

#include "FPGATracklet.hh"
#include "FPGAMemoryBase.hh"
#include "FPGAStub.hh"
#include "L1TStub.hh"

using namespace std;

class FPGAFullMatch:public FPGAMemoryBase{

public:

  FPGAFullMatch(string name, unsigned int iSector, 
		double phimin, double phimax):
    FPGAMemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addMatch(FPGATracklet* tracklet,std::pair<FPGAStub*,L1TStub*> stub) {
    std::pair<FPGATracklet*,std::pair<FPGAStub*,L1TStub*> > tmp(tracklet,stub);
    matches_.push_back(tmp);
  }

  void addMatch(std::pair<FPGATracklet*,std::pair<FPGAStub*,L1TStub*> > match) {
    matches_.push_back(match);
  }

  unsigned int nMatches() const {return matches_.size();}

  FPGATracklet* getFPGATracklet(unsigned int i) const {return matches_[i].first;}

  std::pair<FPGATracklet*,std::pair<FPGAStub*,L1TStub*> > getMatch(unsigned int i) const {return matches_[i];}

  void clean() {
    matches_.clear();
  }


private:

  double phimin_;
  double phimax_;
  std::vector<std::pair<FPGATracklet*,std::pair<FPGAStub*,L1TStub*> > > matches_;

};

#endif
