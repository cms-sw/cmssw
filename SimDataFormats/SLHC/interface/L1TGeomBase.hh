#ifndef L1TGEOMBASE_H
#define L1TGEOMBASE_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
using namespace std;

#include "L1TStub.hh"
#include "L1TTracklet.hh"

class L1TGeomBase{

  friend class L1TDisk;
  friend class L1TBarrel;

private:
  L1TGeomBase(){
  }


public:

  void fitTracks() {
    for(int iSector=0;iSector<NSector_;iSector++){
      for(unsigned int i=0;i<tracklets_[iSector].size();i++){
	if (0) {
	  static ofstream out("nstubs.txt");
	  out << tracklets_[iSector][i].nStubs()<<" "
	      << tracklets_[iSector][i].r()<<" "
	      << tracklets_[iSector][i].z()<<" "
	      << endl;
	}  
	if (tracklets_[iSector][i].nStubs()>3){
	  L1TTrack aTrack(tracklets_[iSector][i]);
	  tracks_[iSector].addTrack(aTrack);
	}
      }
    }
  }

  unsigned int nTracks(int iSector) const {return tracks_[iSector].size();}
  L1TTrack& track(int iSector, unsigned int i) {return tracks_[iSector].get(i);}

  unsigned int nTracklets(int iSector) const {return tracklets_[iSector].size();}
  L1TTracklet& tracklet(int iSector, unsigned int i) {return tracklets_[iSector][i];}

  L1TTracks allTracks(){
    L1TTracks tracks=tracks_[0];
    for (int i=1;i<NSector_;i++){
      tracks.addTracks(tracks_[i]);
    }
    return tracks;
  }

private:

  int NSector_;

  vector<vector<L1TStub> > stubs_;

  vector<vector<L1TTracklet> > tracklets_;

  vector<L1TTracks > tracks_;


};



#endif



