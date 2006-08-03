#ifndef SimDataFormats_SimTkHit_PSimHitContainer_H
#define SimDataFormats_SimTkHit_PSimHitContainer_H

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <vector>

namespace edm {
      typedef std::vector<PSimHit> PSimHitContainer;
} // edm
      typedef edm::Ref<edm::PSimHitContainer> TrackPSimHitRef;         
      typedef edm::RefProd<edm::PSimHitContainer> TrackPSimHitRefProd;
      typedef std::vector<TrackPSimHitRef> TrackPSimHitRefVector;

#endif 

