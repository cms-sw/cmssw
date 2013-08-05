#include "DataFormats/Common/interface/Wrapper.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

namespace {
  struct dictionary {
    std::pair<TrackingParticleRef, TrackPSimHitRef> dummy13;
    edm::Wrapper<std::pair<TrackingParticleRef, TrackPSimHitRef> > dummy14;
    std::vector<std::pair<TrackingParticleRef, TrackPSimHitRef> > dummy07;
    edm::Wrapper<std::vector<std::pair<TrackingParticleRef, TrackPSimHitRef> > > dummy08;
  };
}
