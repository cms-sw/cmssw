#ifndef HitPairGeneratorFromLayerPair_h
#define HitPairGeneratorFromLayerPair_h

#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

class DetLayer;
class TrackingRegion;

class HitPairGeneratorFromLayerPair {

public:

  typedef LayerHitMapCache LayerCacheType;
  typedef SeedingLayerSetsHits::SeedingLayerSet Layers;
  typedef SeedingLayerSetsHits::SeedingLayer Layer;

  HitPairGeneratorFromLayerPair(unsigned int inner,
                                unsigned int outer,
                                LayerCacheType* layerCache,
				unsigned int max=0);

  ~HitPairGeneratorFromLayerPair();

  HitDoublets doublets( const TrackingRegion& reg,
                        const edm::Event & ev,  const edm::EventSetup& es, Layers layers);

  void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs,
                 const edm::Event & ev,  const edm::EventSetup& es, Layers layers);

  Layer innerLayer(const Layers& layers) const { return layers[theInnerLayer]; }
  Layer outerLayer(const Layers& layers) const { return layers[theOuterLayer]; }

private:
  LayerCacheType & theLayerCache;
  const unsigned int theOuterLayer;
  const unsigned int theInnerLayer;
  const unsigned int theMaxElement;
};

#endif
