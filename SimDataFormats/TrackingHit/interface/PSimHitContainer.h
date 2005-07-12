#ifndef SimDataFormats_SimTkHit_PSimHitContainer_H
#define SimDataFormats_SimTkHit_PSimHitContainer_H

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <vector>
#include <map>
#include <string>

namespace edm {
  class PSimHitContainer: public EDProduct {
  public:
      typedef std::vector<PSimHit> PSimHitSingleContainer;
      typedef std::map<std::string,PSimHitSingleContainer> PSimHitMultipleContainer;
      /// insert a Hit for a given layer
      void insertHits(std::string name, PSimHitSingleContainer&);
      void insertHit(std::string name, const PSimHit&);
      void clear(std::string name);
      unsigned int size(std::string name);
  private:
    PSimHitMultipleContainer _data;
  };
} // edm


#endif 

