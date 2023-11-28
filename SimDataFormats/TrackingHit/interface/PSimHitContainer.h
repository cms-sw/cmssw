#ifndef SimDataFormats_SimTkHit_PSimHitContainer_H
#define SimDataFormats_SimTkHit_PSimHitContainer_H

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <vector>
#include "DataFormats/Common/interface/RefToBase.h"

namespace edm {
  typedef std::vector<PSimHit> PSimHitContainer;
}  // namespace edm

typedef edm::Ref<edm::PSimHitContainer> TrackPSimHitRef;
typedef edm::RefProd<edm::PSimHitContainer> TrackPSimHitRefProd;

typedef edm::reftobase::Holder<PSimHit, TrackPSimHitRef> TrackPSimHitRefToBaseHolder;

#endif
