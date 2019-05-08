#ifndef SimTracker_SimHitSelectorFromDB_H
#define SimTracker_SimHitSelectorFromDB_H

// Data
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <map>
#include <vector>

class SimHitSelectorFromDB {
public:
  SimHitSelectorFromDB();
  ~SimHitSelectorFromDB(){};

  //  std::vector<PSimHit> getSimHit(std::unique_ptr<MixCollection<PSimHit>
  //  >&,std::map<uint32_t, std::vector<int> >& );
  std::vector<std::pair<const PSimHit *, int>> getSimHit(std::unique_ptr<MixCollection<PSimHit>> &,
                                                         std::map<uint32_t, std::vector<int>> &);

private:
  //  std::vector<PSimHit> theNewSimHitList;
  std::vector<std::pair<const PSimHit *, int>> theNewSimHitList;
};

#endif
