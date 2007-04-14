#ifndef TkSeedingLayers_SeedingLayer_H
#define TkSeedingLayers_SeedingLayer_H

#include <string>
#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"

class DetLayer;
class TransientTrackingRecHitBuilder;

namespace edm { class Event; class EventSetup; }
namespace ctfseeding {class HitExtractor; }


namespace ctfseeding {

class SeedingLayer {
public:
  enum Side { Barrel = 0, NegEndcap =1,  PosEndcap = 2 }; 
public:
  SeedingLayer( const DetLayer* layer,
                const std::string & name,
                const std::string & hitBuilder,
                const HitExtractor * hitExtractor);

  ~SeedingLayer();

  std::string name() const { return theName; }
  std::vector<SeedingHit> hits(const edm::Event& ev, const edm::EventSetup& es) const;

  bool operator==(const SeedingLayer &s) const { return name()==s.name(); }

  const DetLayer*  detLayer() const { return theLayer; }
  const TransientTrackingRecHitBuilder * hitBuilder(const edm::EventSetup& es) const;
 
private:
  const DetLayer* theLayer;
  std::string theName;
  std::string theTTRHBuilderName;
  const HitExtractor * theHitExtractor;
  mutable const TransientTrackingRecHitBuilder *theTTRHBuilder;
};
}
#endif
