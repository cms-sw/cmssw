#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/src/InnerDeltaPhi.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapLoop.h"
#include "RecoTracker/TkHitPairs/src/RecHitsSortedInPhi.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHit.h"

using namespace GeomDetEnumerators;
using namespace ctfseeding;
using namespace std;

typedef PixelRecoRange<float> Range;
template <class T> T sqr( T t) {return t*t;}



#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

HitPairGeneratorFromLayerPair::HitPairGeneratorFromLayerPair(
    const Layer& inner, const Layer& outer, LayerCacheType* layerCache)
  : theLayerCache(*layerCache), theOuterLayer(outer), theInnerLayer(inner)
{ }

void HitPairGeneratorFromLayerPair::hitPairs(
    const TrackingRegion & region, OrderedHitPairs & result,
    const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  if (theInnerLayer.detLayer()->subDetector() != PixelBarrel &&
      theInnerLayer.detLayer()->location() == barrel ){
    hitPairsWithErrors(region,result,iEvent,iSetup);
    return;
  }

  typedef OrderedHitPair::InnerHit InnerHit;
  typedef OrderedHitPair::OuterHit OuterHit;

  const LayerHitMap & innerHitsMap = theLayerCache(&theInnerLayer, region, iEvent, iSetup);
  if (innerHitsMap.empty()) return;
 
  const LayerHitMap & outerHitsMap = theLayerCache(&theOuterLayer, region, iEvent, iSetup);
  if (outerHitsMap.empty()) return;

  const DetLayer * innerlay = theInnerLayer.detLayer();
  const DetLayer * outerlay = theOuterLayer.detLayer();
  
  float outerHitErrorRPhi = (outerlay->location() == barrel) ?
      TrackingRegionBase::hitErrRPhi(
	  dynamic_cast<const BarrelDetLayer*>(outerlay) )
    : TrackingRegionBase::hitErrRPhi(
          dynamic_cast<const ForwardDetLayer*>(outerlay) ) ;

  InnerDeltaPhi deltaPhi(*innerlay, region, iSetup);

  float rzLayer1, rzLayer2;
  if (innerlay->location() == barrel) {
    const BarrelDetLayer& bl = 
        dynamic_cast<const BarrelDetLayer&>(*innerlay);
    float halfThickness  = bl.surface().bounds().thickness()/2;
    float radius = bl.specificSurface().radius();
    rzLayer1 = radius-halfThickness;
    rzLayer2 = radius+halfThickness;
  } 
  else {
    float halfThickness  = innerlay->surface().bounds().thickness()/2;
    float zLayer = innerlay->position().z() ;
    rzLayer1 = zLayer-halfThickness;
    rzLayer2 = zLayer+halfThickness;
  }

  const SeedingHit * oh;
  LayerHitMapLoop outerHits = outerHitsMap.loop();

  while ( (oh=outerHits.getHit()) ) {
    
    PixelRecoRange<float> phiRange = deltaPhi( oh->r(), oh->phi(), oh->z(), outerHitErrorRPhi);    
    if (phiRange.empty()) continue;

    const HitRZCompatibility *checkRZ = region.checkRZ(&(*innerlay), *oh, iSetup);
    if(!checkRZ) continue;

    LayerHitMapLoop innerHits = innerHitsMap.loop(phiRange, Range(-100.,100.));
    const SeedingHit * ih;
    while ( (ih=innerHits.getHit()) ) {
      float ih_x = ih->r() * cos(ih->phi());
      float ih_y = ih->r() * sin(ih->phi());
      float r_reduced = sqrt( sqr(ih_x-region.origin().x())+sqr(ih_y-region.origin().y()));

      if ( (*checkRZ)( r_reduced, ih->z()) ) {
            result.push_back( OrderedHitPair( *ih, *oh) ); 
        }
    }
    delete checkRZ;
  }
}


void HitPairGeneratorFromLayerPair::
   hitPairsWithErrors( const TrackingRegion& region,
		       OrderedHitPairs & result,
                   const edm::Event & iEvent,
		       const edm::EventSetup& iSetup)
{
  static const TransientTrackingRecHitBuilder * TTRHbuilder = 0;
  static const TrackerGeometry * trackerGeometry = 0;
  if(TTRHbuilder == 0){
    edm::ESHandle<TransientTrackingRecHitBuilder> theBuilderHandle;
    iSetup.get<TransientRecHitRecord>().get("WithoutRefit",theBuilderHandle);
    TTRHbuilder = theBuilderHandle.product();
  }
  if (!trackerGeometry) {
    edm::ESHandle<TrackerGeometry> tracker;
    iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
    trackerGeometry = tracker.product();
  }


  typedef OrderedHitPair::InnerHit InnerHit;
  typedef OrderedHitPair::OuterHit OuterHit;

   vector<SeedingHit> oSHits(region.hits(iEvent,iSetup,&theOuterLayer));
   vector<const TrackingRecHit*> outerHits;
   typedef vector<SeedingHit>::const_iterator ISH;
   for (ISH it=oSHits.begin(); it != oSHits.end(); ++it) outerHits.push_back( *it);
   vector<SeedingHit> iSHits(theInnerLayer.hits(iEvent,iSetup));
   vector<const TrackingRecHit*> innerHits;
   for (ISH it=iSHits.begin(); it != iSHits.end(); ++it) innerHits.push_back( *it);
   RecHitsSortedInPhi innerSortedHits(innerHits,trackerGeometry);
   
				   
  InnerDeltaPhi deltaPhi(
      *(theInnerLayer.detLayer()), region, iSetup);

  typedef vector<const TrackingRecHit*>::const_iterator  HI;
  float nSigmaRZ = sqrt(12.);
  float nSigmaPhi = 3.;
  for (HI oh=outerHits.begin(); oh!= outerHits.end(); oh++) {
    TransientTrackingRecHit::RecHitPointer recHit = TTRHbuilder->build(*oh);
    GlobalPoint hitPos = recHit->globalPosition();
    float phiErr = nSigmaPhi * sqrt(recHit->globalPositionError().phierr(hitPos)); 
    float dphi = deltaPhi( hitPos.perp(), hitPos.z(), hitPos.perp()*phiErr);   
    float phiHit = hitPos.phi();
    std::cout<<"hit pairs generator dphi:"<<dphi<<std::endl;
    vector<const TrackingRecHit*> innerCandid = innerSortedHits.hits(phiHit-dphi,phiHit+dphi);
    const HitRZCompatibility *checkRZ = region.checkRZ(theInnerLayer.detLayer(), *oh,iSetup);
    if(!checkRZ) continue;

    for (HI ih = innerCandid.begin(); ih != innerCandid.end(); ih++) {
      TransientTrackingRecHit::RecHitPointer recHit = TTRHbuilder->build(&(**ih));
      GlobalPoint innPos = recHit->globalPosition();
      Range allowed = checkRZ->range(innPos.perp());
      Range hitRZ;
      if (theInnerLayer.detLayer()->location() == barrel) {
        float zErr = nSigmaRZ * sqrt(recHit->globalPositionError().czz());
        hitRZ = Range(innPos.z()-zErr, innPos.z()+zErr);
      } else {
        float rErr = nSigmaRZ * sqrt(recHit->globalPositionError().rerr(innPos));
        hitRZ = Range(innPos.perp()-rErr, innPos.perp()+rErr);
      }
      Range crossRange = allowed.intersection(hitRZ);
      if (! crossRange.empty() ) {
        result.push_back( OrderedHitPair( SeedingHit(*ih,theInnerLayer,iSetup), SeedingHit(*oh,theOuterLayer, iSetup) ) );
      }
    } 
    delete checkRZ;
  }
}

