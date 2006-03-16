#ifndef HitPairGeneratorFromLayerPair_h
#define HitPairGeneratorFromLayerPair_h

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "FWCore/Framework/interface/EventSetup.h"
class DetLayer;
class TrackingRegion;
class LayerWithHits;
class HitPairGeneratorFromLayerPair : public HitPairGenerator {

public:

  typedef CombinedHitPairGenerator::LayerCacheType       LayerCacheType;
 



  HitPairGeneratorFromLayerPair( 
				const LayerWithHits* inner, 
				const LayerWithHits* outer, 
				LayerCacheType* layerCache, 
				const edm::EventSetup& iSetup)
    : theLayerCache(*layerCache), theOuterLayer(outer), theInnerLayer(inner) { 
    //MP
    // innerlay= inner->layer();
  }

  virtual ~HitPairGeneratorFromLayerPair() { }

  virtual OrderedHitPairs hitPairs( const TrackingRegion& region,const edm::EventSetup& iSetup ) {
    return HitPairGenerator::hitPairs(region, iSetup);
  }
  virtual void hitPairs( const TrackingRegion& ar, OrderedHitPairs & ap,const edm::EventSetup& iSetup);

  virtual HitPairGeneratorFromLayerPair* clone() const {
    return new HitPairGeneratorFromLayerPair(*this);
  }

  const LayerWithHits* innerLayer() const { return theInnerLayer; }
  const LayerWithHits* outerLayer() const { return theOuterLayer; }

private:

  // all data members are "shallow copy"
  LayerCacheType & theLayerCache;
  const LayerWithHits* theOuterLayer;  
  const LayerWithHits* theInnerLayer; 
  const PixelBarrelLayer* innerlay;
  const PixelBarrelLayer* outerlay;

};

#endif
