#ifndef CosmicHitPairGeneratorFromLayerPair_h
#define CosmicHitPairGeneratorFromLayerPair_h

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
class DetLayer;
class TrackingRegion;
class LayerWithHits;
 class CompareHitPairsY {
 public:
   CompareHitPairsY(const edm::EventSetup& iSetup){    

     iSetup.get<TrackerDigiGeometryRecord>().get(tracker);};
   bool operator()( OrderedHitPair h1,
		    OrderedHitPair h2)
   {      
     GlobalPoint in1p=tracker->idToDet(h1.inner()->geographicalId())->surface().toGlobal(h1.inner()->localPosition());
     GlobalPoint in2p=tracker->idToDet(h2.inner()->geographicalId())->surface().toGlobal(h2.inner()->localPosition());
     GlobalPoint ou1p=tracker->idToDet(h1.outer()->geographicalId())->surface().toGlobal(h1.outer()->localPosition());
     GlobalPoint ou2p=tracker->idToDet(h2.outer()->geographicalId())->surface().toGlobal(h2.outer()->localPosition());
     if (ou1p.y()*ou2p.y()<0) return ou1p.y()>ou2p.y();
     else{
       float dist1=100*abs(ou1p.z()-in1p.z())-abs(ou1p.y())-0.1*abs(in1p.y());
       float dist2=100*abs(ou2p.z()-in2p.z())-abs(ou2p.y())-0.1*abs(in2p.y());
       return dist1 < dist2;
     }
   }
 private:
   edm::ESHandle<TrackerGeometry> tracker;
 };
class CosmicHitPairGeneratorFromLayerPair : public HitPairGenerator {

public:


  CosmicHitPairGeneratorFromLayerPair( 
				const LayerWithHits* inner, 
				const LayerWithHits* outer, 
				const edm::EventSetup& iSetup);
  virtual ~CosmicHitPairGeneratorFromLayerPair() { }

  virtual OrderedHitPairs hitPairs( const TrackingRegion& region,const edm::EventSetup& iSetup ) {
    return HitPairGenerator::hitPairs(region, iSetup);
  }
  virtual void hitPairs( const TrackingRegion& ar, OrderedHitPairs & ap,const edm::EventSetup& iSetup);

  virtual CosmicHitPairGeneratorFromLayerPair* clone() const {
    return new CosmicHitPairGeneratorFromLayerPair(*this);
  }

  const LayerWithHits* innerLayer() const { return theInnerLayer; }
  const LayerWithHits* outerLayer() const { return theOuterLayer; }

private:
  const TransientTrackingRecHitBuilder * TTRHbuilder;
  const TrackerGeometry* trackerGeometry;
  const LayerWithHits* theOuterLayer;  
  const LayerWithHits* theInnerLayer; 
  const DetLayer* innerlay;
  const DetLayer* outerlay;

};

#endif
