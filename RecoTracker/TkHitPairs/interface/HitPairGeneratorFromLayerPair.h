#ifndef HitPairGeneratorFromLayerPair_h
#define HitPairGeneratorFromLayerPair_h

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"

class DetLayer;
class TrackingRegion;

class HitPairGeneratorFromLayerPair : public HitPairGenerator {

public:

  typedef CombinedHitPairGenerator::LayerCacheType       LayerCacheType;
  typedef SeedingLayerSetsHits::SeedingLayerSet Layers;
  typedef SeedingLayerSetsHits::SeedingLayer Layer;

  HitPairGeneratorFromLayerPair(unsigned int inner,
                                unsigned int outer,
                                LayerCacheType* layerCache,
				unsigned int nSize=30000,
				unsigned int max=0);

  virtual ~HitPairGeneratorFromLayerPair() { }

  void setSeedingLayers(Layers layers) override { theSeedingLayers = layers; }

  virtual HitDoublets doublets( const TrackingRegion& reg,
			     const edm::Event & ev,  const edm::EventSetup& es);

  virtual void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs,
      const edm::Event & ev,  const edm::EventSetup& es);

  virtual HitPairGeneratorFromLayerPair* clone() const {
    return new HitPairGeneratorFromLayerPair(*this);
  }

  Layer innerLayer() const { return theSeedingLayers[theInnerLayer]; }
  Layer outerLayer() const { return theSeedingLayers[theOuterLayer]; }

private:
  LayerCacheType & theLayerCache;
  Layers theSeedingLayers;
  const unsigned int theOuterLayer;
  const unsigned int theInnerLayer;
};

#endif
