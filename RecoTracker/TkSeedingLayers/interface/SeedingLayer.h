#ifndef TkSeedingLayers_SeedingLayer_H
#define TkSeedingLayers_SeedingLayer_H

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

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

  SeedingLayer( const std::string & name,
                const DetLayer* layer,
                const TransientTrackingRecHitBuilder * hitBuilder,
                const HitExtractor * hitExtractor,  
                bool usePredefinedErrors = false, float hitErrorRZ = 0., float hitErrorRPhi=0.);

  std::string name() const;

  void hits(const edm::Event& ev, const edm::EventSetup& es, Hits &) const;
  Hits hits(const edm::Event& ev, const edm::EventSetup& es) const;

  bool operator==(const SeedingLayer &s) const { return name()==s.name(); }

  const DetLayer*  detLayer() const;
  
  const TransientTrackingRecHitBuilder * hitBuilder() const;

  bool hasPredefinedHitErrors() const;
  float predefinedHitErrorRZ() const;
  float predefinedHitErrorRPhi() const;
 
private:
  class SeedingLayerImpl;
  boost::shared_ptr<SeedingLayerImpl> theImpl;
};

}
#endif
