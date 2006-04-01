#include "RecoTracker/TkHitPairs/interface/CosmicHitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

//#include "CommonDet/BasicDet/interface/RecHit.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
//#include "RecoTracker/TkHitPairs/interface/LayerHitsCache.h"
#include "RecoTracker/TkHitPairs/interface/InnerDeltaPhi.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapLoop.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

typedef PixelRecoRange<float> Range;

void CosmicHitPairGeneratorFromLayerPair::hitPairs(
  const TrackingRegion & region, OrderedHitPairs & result,
  const edm::EventSetup& iSetup)
{
//  static int NSee = 0; static int Ntry = 0; static int Nacc = 0;


  typedef OrderedHitPair::InnerHit InnerHit;
  typedef OrderedHitPair::OuterHit OuterHit;


  const LayerHitMap & innerHitsMap = theLayerCache(theInnerLayer, region,iSetup);
  if (innerHitsMap.empty()) return;
 
  const LayerHitMap & outerHitsMap = theLayerCache(theOuterLayer, region,iSetup);
  if (outerHitsMap.empty()) return;


  vector<OrderedHitPair> allthepairs;
  const TkHitPairsCachedHit * oh;
  LayerHitMapLoop outerHits = outerHitsMap.loop();
//  static TimingReport::Item * theTimer1 =
//        PixelRecoUtilities::initTiming("--- outerHitloop ",1);
//  TimeMe tm1( *theTimer1, false);

  while ( (oh=outerHits.getHit()) ) {
    LayerHitMapLoop innerHits = innerHitsMap.loop();
    const TkHitPairsCachedHit * ih;
 
    while ( (ih=innerHits.getHit()) ) {
     float differenza =ih->z()-oh->z();
     float inny=ih->r()*sin(ih->phi());
     float outy=oh->r()*sin(oh->phi());
     if( (differenza<30)&&((inny-outy)<30)&&(inny<0)) 
       allthepairs.push_back( OrderedHitPair(ih->RecHit(), oh->RecHit()));
    }

  }
  stable_sort(allthepairs.begin(),allthepairs.end(),CompareHitPairsY(iSetup));
  
  if (allthepairs.size()>0)   result.push_back(allthepairs[0]);  
    
}
  


