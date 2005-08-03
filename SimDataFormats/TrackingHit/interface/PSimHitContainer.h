#ifndef SimDataFormats_SimTkHit_PSimHitContainer_H
#define SimDataFormats_SimTkHit_PSimHitContainer_H

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <vector>
#include <map>
#include <string>

namespace edm {
  class PSimHitContainer {
  public:
      typedef std::vector<PSimHit> PSimHitSingleContainer;
      /// insert a Hit for a given layer
      void insertHits(PSimHitSingleContainer&);
      void insertHit(const PSimHit&);
      void clear();
      unsigned int size();
  private:
    PSimHitSingleContainer _data;
  };
} // edm


#endif 

