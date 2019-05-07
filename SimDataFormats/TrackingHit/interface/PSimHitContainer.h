#ifndef SimDataFormats_SimTkHit_PSimHitContainer_H
#define SimDataFormats_SimTkHit_PSimHitContainer_H

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include <vector>

namespace edm {
typedef std::vector<PSimHit> PSimHitContainer;
} // namespace edm

typedef edm::Ref<edm::PSimHitContainer> TrackPSimHitRef;
typedef edm::RefProd<edm::PSimHitContainer> TrackPSimHitRefProd;

typedef std::vector<edm::RefToBase<PSimHit>> TrackPSimHitRefToBaseVector;
typedef edm::RefToBase<PSimHit> TrackPSimHitRefToBase;
typedef std::vector<edm::RefToBase<PSimHit>> TrackPSimHitRefToBaseVector;
typedef edm::reftobase::Holder<PSimHit, TrackPSimHitRef>
    TrackPSimHitRefToBaseHolder;

#endif
