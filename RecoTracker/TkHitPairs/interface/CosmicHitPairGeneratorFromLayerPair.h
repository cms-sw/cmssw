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
   bool operator()( OrderedHitPair h1,
		    OrderedHitPair h2)
   {      
     GlobalPoint in1p=tracker->idToDet(h1.inner().RecHit()->geographicalId())->surface().toGlobal(h1.inner().RecHit()->localPosition());
     GlobalPoint in2p=tracker->idToDet(h2.inner().RecHit()->geographicalId())->surface().toGlobal(h2.inner().RecHit()->localPosition());
     GlobalPoint ou1p=tracker->idToDet(h1.outer().RecHit()->geographicalId())->surface().toGlobal(h1.outer().RecHit()->localPosition());
     GlobalPoint ou2p=tracker->idToDet(h2.outer().RecHit()->geographicalId())->surface().toGlobal(h2.outer().RecHit()->localPosition());
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
