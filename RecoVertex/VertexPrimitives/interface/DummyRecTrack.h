#ifndef _DummyRecTrack_H_
#define _DummyRecTrack_H_

#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


class DummyRecTrack {

public:

  DummyRecTrack(const TrackCharge& aCharge, const TrajectoryStateOnSurface aTsos): 
    theCharge(aCharge), theTsos(aTsos){}

  ~DummyRecTrack() {}
  
  TrackCharge charge() const {return theCharge;}
  bool operator== (const DummyRecTrack & a) const {return true;}
  TrajectoryStateOnSurface impactPointState() const {return theTsos;} 

private:

  TrackCharge theCharge;
  TrajectoryStateOnSurface theTsos;

};

#endif
