#ifndef RecoTracker_TkSeedingLayers_HitExtractor_H
#define RecoTracker_TkSeedingLayers_HitExtractor_H

#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"
namespace edm { class Event; class EventSetup; }

namespace ctfseeding {
class HitExtractor {
public:
  virtual ~HitExtractor(){}
  virtual std::vector<SeedingHit> hits(const edm::Event& , const edm::EventSetup& ) const =0;
};
}

#endif
