#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {
    PSimHit dummy444;
    edm::PSimHitContainer sdummy777;
    edm::Wrapper<edm::PSimHitContainer> dummy7777;
    std::vector<const PSimHit*> dummyvcp;

    TrackPSimHitRef r7;
    TrackPSimHitRefProd rp7; 

    TrackPSimHitRefToBase rb7;
    TrackPSimHitRefToBaseVector rbv7;
    TrackPSimHitRefToBaseHolder rbh7; 

  };
}
