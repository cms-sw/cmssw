#ifndef L1TTRACKLETS_H
#define L1TTRACKLETS_H

#include <iostream>
#include <assert.h>

#include "L1TTracklet.hh"


using namespace std;

//This is the number of strips in rphi and in z for a module.
//This should be in the header of the ASCII file, but for now
//just hardcoded here.



class L1TTracklets{

public:

  L1TTracklets(){

  }

  void addTracklet(const L1TTracklet& aTracklet){
    tracklets_.push_back(aTracklet);
  }

  void print() {

    for (unsigned int i=0;i<tracklets_.size();i++){
      tracklets_[i].print();
    }

  }
  

  unsigned int size() { return tracklets_.size(); }

  L1TTracklet& get(unsigned int i) { return tracklets_[i];}

  void clean(){
    tracklets_.clear();
  }

private:

  vector<L1TTracklet> tracklets_;

};



#endif



