#ifndef SimTracker_SimHitSelectorFromDB_H
#define SimTracker_SimHitSelectorFromDB_H

//Data
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "CLHEP/Random/RandFlat.h"


class SimHitSelectorFromDB {

 public:
  SimHitSelectorFromDB();
  ~SimHitSelectorFromDB(){};
  
  std::vector<PSimHit> getSimHit(std::auto_ptr<MixCollection<PSimHit> >&,std::map<uint32_t, std::vector<int> >& );
 private:
  std::vector<PSimHit> theNewSimHitList;

};

#endif
