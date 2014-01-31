#ifndef TkSeedingLayers_SeedingLayer_H
#define TkSeedingLayers_SeedingLayer_H

#include <string>
#include <vector>
#include <memory>

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class DetLayer;
class TransientTrackingRecHitBuilder;

namespace edm { class Event; class EventSetup; }
namespace ctfseeding {class HitExtractor; }

namespace ctfseeding {

class SeedingLayer {
public:
  enum Side { Barrel = 0, NegEndcap =1,  PosEndcap = 2 }; 
public:
  typedef  std::vector<TransientTrackingRecHit::ConstRecHitPointer> Hits;
  
  SeedingLayer(){}

  SeedingLayer( const std::string & name, int seqNum,
                const DetLayer* layer,
                const TransientTrackingRecHitBuilder * hitBuilder,
                const HitExtractor * hitExtractor);

  std::string name() const;
  int seqNum() const;

  void hits(const edm::Event& ev, const edm::EventSetup& es, Hits &) const;
  Hits hits(const edm::Event& ev, const edm::EventSetup& es) const;

  bool operator==(const SeedingLayer &s) const { return name()==s.name(); }

  const DetLayer*  detLayer() const;
  
  const TransientTrackingRecHitBuilder * hitBuilder() const;

private:
  class SeedingLayerImpl;
  std::shared_ptr<SeedingLayerImpl> theImpl;
};

}
#endif
