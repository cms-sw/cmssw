#ifndef TkSeedingLayers_SeedingLayer_H
#define TkSeedingLayers_SeedingLayer_H

#include <string>
#include <vector>
#include <memory>

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackingRecHit/interface/mayown_ptr.h"


class DetLayer;
class TransientTrackingRecHitBuilder;
class TkTransientTrackingRecHitBuilder;

namespace edm { class Event; class EventSetup; }
namespace ctfseeding {class HitExtractor; }

namespace ctfseeding {

class SeedingLayer {
public:
  enum Side { Barrel = 0, NegEndcap =1,  PosEndcap = 2 }; 
public:
  using TkHit = BaseTrackerRecHit;
  using TkHitRef = BaseTrackerRecHit const &;
  using HitPointer = mayown_ptr<BaseTrackerRecHit>;
  using Hits=std::vector<HitPointer>;
  
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
  
  const TkTransientTrackingRecHitBuilder * hitBuilder() const;

private:
  class SeedingLayerImpl;
  std::shared_ptr<SeedingLayerImpl> theImpl;
};

}
#endif
