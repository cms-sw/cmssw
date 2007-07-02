#ifndef TkSeedingLayerImpls_SeedingLayerImpl_H
#define TkSeedingLayerImpls_SeedingLayerImpl_H

#include <string>
#include <vector>
#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"

class DetLayer;
class TransientTrackingRecHitBuilder;

namespace edm { class Event; class EventSetup; }
namespace ctfseeding {class HitExtractor; class SeedingLayer; }


namespace ctfseeding {

class SeedingLayerImpl {
public:
  SeedingLayerImpl( 
                const std::string & name,
                const DetLayer* layer,
                const TransientTrackingRecHitBuilder * hitBuilder,
                const HitExtractor * hitExtractor);

  SeedingLayerImpl( 
                const std::string & name,
                const DetLayer* layer,
                const TransientTrackingRecHitBuilder * hitBuilder,
                const HitExtractor * hitExtractor, float hitErrorRZ, float hitErrorRPhi);

  ~SeedingLayerImpl();

  std::string name() const { return theName; }
  std::vector<SeedingHit> hits(const SeedingLayer &, const edm::Event& ev, const edm::EventSetup& es) const;

  const DetLayer*  detLayer() const { return theLayer; }
  const TransientTrackingRecHitBuilder * hitBuilder() const { return theTTRHBuilder; }

  bool  hasPredefinedHitErrors() const { return theHasPredefinedHitErrors; }
  float predefinedHitErrorRZ() const { return thePredefinedHitErrorRZ; }
  float predefinedHitErrorRPhi() const { return thePredefinedHitErrorRPhi; }

private:
  SeedingLayerImpl(const SeedingLayerImpl &);
private:
  std::string theName;
  const DetLayer* theLayer;
  const TransientTrackingRecHitBuilder *theTTRHBuilder;
  const HitExtractor * theHitExtractor;
  bool theHasPredefinedHitErrors;
  float thePredefinedHitErrorRZ, thePredefinedHitErrorRPhi;
};
}
#endif
