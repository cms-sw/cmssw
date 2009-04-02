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
    const Layer& inner, const Layer& outer, LayerCacheType* layerCache, unsigned int nSize)
  : HitPairGenerator(nSize),
    theLayerCache(*layerCache), theOuterLayer(outer), theInnerLayer(inner)
{ }

void HitPairGeneratorFromLayerPair::hitPairs(
    const TrackingRegion & region, OrderedHitPairs & result,
    const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  if (theInnerLayer.detLayer()->subDetector() != PixelBarrel &&
      theInnerLayer.detLayer()->subDetector() != PixelEndcap ){
    hitPairsWithErrors(region,result,iEvent,iSetup);
    return;
  }

  typedef OrderedHitPair::InnerRecHit InnerHit;
  typedef OrderedHitPair::OuterRecHit OuterHit;
  typedef RecHitsSortedInPhi::Hit Hit;

  const RecHitsSortedInPhi & innerHitsMap = theLayerCache(&theInnerLayer, region, iEvent, iSetup);
  if (innerHitsMap.empty()) return;
 
  const RecHitsSortedInPhi& outerHitsMap = theLayerCache(&theOuterLayer, region, iEvent, iSetup);
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

  vector<Hit> outerHits = outerHitsMap.hits();
  typedef vector<Hit>::const_iterator IT;

  for (IT oh = outerHits.begin(), oeh = outerHits.end(); oh < oeh; ++oh) { 
   
    GlobalPoint oPos = (*oh)->globalPosition();  
    PixelRecoRange<float> phiRange = deltaPhi( oPos.perp(), oPos.phi(), oPos.z(), outerHitErrorRPhi);    

    if (phiRange.empty()) continue;

    const HitRZCompatibility *checkRZ = region.checkRZ(&(*innerlay), (*oh)->hit(), iSetup);
    if(!checkRZ) continue;

    vector<Hit> innerHits;
    innerHitsMap.hits(phiRange.min(), phiRange.max(), innerHits);
    for (IT ih=innerHits.begin(), ieh = innerHits.end(); ih < ieh; ++ih) {  
      GlobalPoint iPos = (*ih)->globalPosition(); 
      float ih_x = iPos.x();
      float ih_y = iPos.y();
      float r_reduced = sqrt( sqr(ih_x-region.origin().x())+sqr(ih_y-region.origin().y()));

      if ( (*checkRZ)( r_reduced, iPos.z()) ) {
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
  const TrackerGeometry * trackerGeometry = 0;
  
  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  trackerGeometry = tracker.product();

  typedef OrderedHitPair::InnerRecHit InnerHit;
  typedef OrderedHitPair::OuterRecHit OuterHit;
  typedef TransientTrackingRecHit::ConstRecHitPointer Hit;

  vector<Hit> outerHits( region.hits(iEvent,iSetup,&theOuterLayer));
  vector<Hit> innerHits( region.hits(iEvent,iSetup,&theInnerLayer));
  RecHitsSortedInPhi innerSortedHits(innerHits);
   
  InnerDeltaPhi deltaPhi( *(theInnerLayer.detLayer()), region, iSetup);

  typedef vector<Hit>::const_iterator  HI;
  float nSigmaRZ = sqrt(12.);
  float nSigmaPhi = 3.;
  for (HI oh=outerHits.begin(); oh!= outerHits.end(); oh++) {
    GlobalPoint hitPos = (*oh)->globalPosition();
    float phiErr = nSigmaPhi * sqrt( (*oh)->globalPositionError().phierr(hitPos)); 
    float dphi = deltaPhi( hitPos.perp(), hitPos.z(), hitPos.perp()*phiErr);   
    float phiHit = hitPos.phi();
    vector<Hit> innerCandid = innerSortedHits.hits(phiHit-dphi,phiHit+dphi);
    const HitRZCompatibility *checkRZ = region.checkRZ(theInnerLayer.detLayer(), (*oh)->hit(),iSetup);
    if(!checkRZ) continue;

    for (HI ih = innerCandid.begin(); ih != innerCandid.end(); ih++) {
      GlobalPoint innPos = (*ih)->globalPosition();
      Range allowed;
      Range hitRZ;
      if (theInnerLayer.detLayer()->location() == barrel) {
	allowed = checkRZ->range(innPos.perp());
        float zErr = nSigmaRZ * sqrt( (*ih)->globalPositionError().czz());
        hitRZ = Range(innPos.z()-zErr, innPos.z()+zErr);
      } else {
	allowed = checkRZ->range(innPos.z());
        float rErr = nSigmaRZ * sqrt( (*ih)->globalPositionError().rerr(innPos));
        hitRZ = Range(innPos.perp()-rErr, innPos.perp()+rErr);
      }
      Range crossRange = allowed.intersection(hitRZ);
      if (! crossRange.empty() ) {
        result.push_back( OrderedHitPair( *ih, *oh) );
      }
    } 
    delete checkRZ;
  }
}

