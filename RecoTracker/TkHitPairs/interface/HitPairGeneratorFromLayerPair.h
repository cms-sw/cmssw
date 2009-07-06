#ifndef HitPairGeneratorFromLayerPair_h
#define HitPairGeneratorFromLayerPair_h

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "FWCore/Framework/interface/EventSetup.h"

class DetLayer;
class TrackingRegion;

class HitPairGeneratorFromLayerPair : public HitPairGenerator {

public:

  typedef CombinedHitPairGenerator::LayerCacheType       LayerCacheType;
  typedef ctfseeding::SeedingLayer Layer;
 
  HitPairGeneratorFromLayerPair(const Layer& inner, const Layer& outer, LayerCacheType* layerCache, unsigned int nSize=30000);

  virtual ~HitPairGeneratorFromLayerPair() { }

  virtual void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs, 
      const edm::Event & ev,  const edm::EventSetup& es);

  virtual HitPairGeneratorFromLayerPair* clone() const {
    return new HitPairGeneratorFromLayerPair(*this);
  }

  const Layer & innerLayer() const { return theInnerLayer; }
  const Layer & outerLayer() const { return theOuterLayer; }

private:
  LayerCacheType & theLayerCache;
  Layer theOuterLayer;  
  Layer theInnerLayer; 
};

#endif
