#ifndef CosmicHitPairGeneratorFromLayerPair_h
#define CosmicHitPairGeneratorFromLayerPair_h

#include "RecoTracker/TkHitPairs/interface/HitPairGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerWithHits.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
class DetLayer;
class TrackingRegion;
class LayerWithHits;
 class CompareHitPairsY {
 public:
   CompareHitPairsY(const edm::EventSetup& iSetup){    

     iSetup.get<TrackerDigiGeometryRecord>().get(tracker);};
   bool operator()( const OrderedHitPair& h1,
		    const OrderedHitPair& h2)
   {      
     const TrackingRecHit * trh1i = h1.inner()->hit();
     const TrackingRecHit * trh2i = h2.inner()->hit();
     const TrackingRecHit * trh1o = h1.outer()->hit();
     const TrackingRecHit * trh2o = h2.outer()->hit();
     GlobalPoint in1p=tracker->idToDet(trh1i->geographicalId())->surface().toGlobal(trh1i->localPosition());
     GlobalPoint in2p=tracker->idToDet(trh2i->geographicalId())->surface().toGlobal(trh2i->localPosition());
     GlobalPoint ou1p=tracker->idToDet(trh1o->geographicalId())->surface().toGlobal(trh1o->localPosition());
     GlobalPoint ou2p=tracker->idToDet(trh2o->geographicalId())->surface().toGlobal(trh2o->localPosition());
     if (ou1p.y()*ou2p.y()<0) return ou1p.y()>ou2p.y();
     else{
       float dist1=100*std::abs(ou1p.z()-in1p.z())-std::abs(ou1p.y())-0.1*std::abs(in1p.y());
       float dist2=100*std::abs(ou2p.z()-in2p.z())-std::abs(ou2p.y())-0.1*std::abs(in2p.y());
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

  void setSeedingLayers(SeedingLayerSetsHits::SeedingLayerSet layers) override {}

//  virtual OrderedHitPairs hitPairs( const TrackingRegion& region,const edm::EventSetup& iSetup ) {
//    return HitPairGenerator::hitPairs(region, iSetup);
//  }
  virtual void hitPairs( const TrackingRegion& ar, OrderedHitPairs & ap, const edm::EventSetup& iSetup);

  virtual void hitPairs( const TrackingRegion& ar, OrderedHitPairs & ap, const edm::Event & ev, const edm::EventSetup& iSetup) {}

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
