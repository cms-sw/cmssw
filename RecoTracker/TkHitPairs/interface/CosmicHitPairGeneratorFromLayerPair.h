#ifndef CosmicHitPairGeneratorFromLayerPair_h
#define CosmicHitPairGeneratorFromLayerPair_h

#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
class DetLayer;
class TrackingRegion;
class LayerWithHits;

class CosmicHitPairGeneratorFromLayerPair {
public:
  CosmicHitPairGeneratorFromLayerPair(const LayerWithHits* inner, const LayerWithHits* outer, const TrackerGeometry&);
  ~CosmicHitPairGeneratorFromLayerPair();

  //  virtual OrderedHitPairs hitPairs( const TrackingRegion& region,const edm::EventSetup& iSetup ) {
  //    return HitPairGenerator::hitPairs(region, iSetup);
  //  }
  void hitPairs(const TrackingRegion& ar, OrderedHitPairs& ap);

  const LayerWithHits* innerLayer() const { return theInnerLayer; }
  const LayerWithHits* outerLayer() const { return theOuterLayer; }

private:
  const TrackerGeometry* trackerGeometry;
  const LayerWithHits* theOuterLayer;
  const LayerWithHits* theInnerLayer;
  const DetLayer* innerlay;
  const DetLayer* outerlay;
};

#endif
