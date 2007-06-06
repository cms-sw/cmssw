#ifndef HitPairGeneratorFromLayerPairBC_h
#define HitPairGeneratorFromLayerPairBC_h

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGeneratorBC.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"


class DetLayer;
class TrackingRegion;
class LayerWithHits;
class HitPairGeneratorFromLayerPairBC : public HitPairGenerator {

public:

  typedef CombinedHitPairGeneratorBC::LayerCacheType       LayerCacheType;
 



  HitPairGeneratorFromLayerPairBC(const LayerWithHits* inner, 
				const LayerWithHits* outer, 
				LayerCacheType* layerCache, 
				const edm::EventSetup& iSetup);

  virtual ~HitPairGeneratorFromLayerPairBC() { }

  virtual void hitPairs( const TrackingRegion& ar, OrderedHitPairs & ap, const edm::EventSetup& iSetup);

  virtual void hitPairs( const TrackingRegion& ar, OrderedHitPairs & ap, const edm::Event& ev, const edm::EventSetup& iSetup) {}

  virtual HitPairGeneratorFromLayerPairBC* clone() const {
    return new HitPairGeneratorFromLayerPairBC(*this);
  }

  const LayerWithHits* innerLayer() const { return theInnerLayer; }
  const LayerWithHits* outerLayer() const { return theOuterLayer; }

private:
  void hitPairsWithErrors( const TrackingRegion& ar,
			   OrderedHitPairs & ap,
			   const edm::EventSetup& iSetup);


  const TransientTrackingRecHitBuilder * TTRHbuilder;
  const TrackerGeometry* trackerGeometry;
  LayerCacheType & theLayerCache;
  const LayerWithHits* theOuterLayer;  
  const LayerWithHits* theInnerLayer; 
  const DetLayer* innerlay;
  const DetLayer* outerlay;

};

#endif
