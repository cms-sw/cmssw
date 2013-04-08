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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayer.h"

using namespace GeomDetEnumerators;
using namespace ctfseeding;
using namespace std;

typedef PixelRecoRange<float> Range;
template<class T> inline T sqr( T t) {return t*t;}



#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

HitPairGeneratorFromLayerPair::HitPairGeneratorFromLayerPair(
							     const Layer& inner, 
							     const Layer& outer, 
							     LayerCacheType* layerCache,
							     unsigned int nSize,
							     unsigned int max)
  : HitPairGenerator(nSize),
    theLayerCache(*layerCache), theOuterLayer(outer), theInnerLayer(inner)
{
  theMaxElement=max;
}

void HitPairGeneratorFromLayerPair::hitPairs(
    const TrackingRegion & region, OrderedHitPairs & result,
    const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  typedef OrderedHitPair::InnerRecHit InnerHit;
  typedef OrderedHitPair::OuterRecHit OuterHit;
  typedef RecHitsSortedInPhi::Hit Hit;

  const RecHitsSortedInPhi & innerHitsMap = theLayerCache(&theInnerLayer, region, iEvent, iSetup);
  if (innerHitsMap.empty()) return;
 
  const RecHitsSortedInPhi& outerHitsMap = theLayerCache(&theOuterLayer, region, iEvent, iSetup);
  if (outerHitsMap.empty()) return;

  InnerDeltaPhi deltaPhi(*theInnerLayer.detLayer(), region, iSetup);

  RecHitsSortedInPhi::Range outerHits = outerHitsMap.all();

  static const float nSigmaRZ = std::sqrt(12.f);
  static const float nSigmaPhi = 3.f;
  vector<Hit> innerHits;
  for (RecHitsSortedInPhi::HitIter oh = outerHits.first; oh!= outerHits.second; ++oh) { 
    Hit ohit = (*oh).hit();
    GlobalPoint oPos = ohit->globalPosition();  
    PixelRecoRange<float> phiRange = deltaPhi( oPos.perp(), oPos.phi(), oPos.z(), nSigmaPhi*(ohit->errorGlobalRPhi()));    

    if (phiRange.empty()) continue;

    const HitRZCompatibility *checkRZ = region.checkRZ(theInnerLayer.detLayer(), ohit, iSetup);
    if(!checkRZ) continue;

    innerHits.clear();
    innerHitsMap.hits(phiRange.min(), phiRange.max(), innerHits);
    LogDebug("HitPairGeneratorFromLayerPair")<<
      "preparing for combination of: "<<innerHits.size()<<" inner and: "<<outerHits.second-outerHits.first<<" outter";
    for ( vector<Hit>::const_iterator ih=innerHits.begin(), ieh = innerHits.end(); ih < ieh; ++ih) {  
      GlobalPoint innPos = (*ih)->globalPosition();
      float r_reduced = std::sqrt( sqr(innPos.x()-region.origin().x())+sqr(innPos.y()-region.origin().y()));
      Range allowed;
      Range hitRZ;
      if (theInnerLayer.detLayer()->location() == barrel) {
        allowed = checkRZ->range(r_reduced);
        float zErr = nSigmaRZ * (*ih)->errorGlobalZ();
        hitRZ = Range(innPos.z()-zErr, innPos.z()+zErr);
      } else {
        allowed = checkRZ->range(innPos.z());
        float rErr = nSigmaRZ * (*ih)->errorGlobalR();
        hitRZ = Range(r_reduced-rErr, r_reduced+rErr);
      }
      Range crossRange = allowed.intersection(hitRZ);
      if (! crossRange.empty() ) {
	if (theMaxElement!=0 && result.size() >= theMaxElement){
	  result.clear();
	  edm::LogError("TooManyPairs")<<"number of pairs exceed maximum, no pairs produced";
	  delete checkRZ;
	  return;
	}
        result.push_back( OrderedHitPair( *ih, ohit) );
      }
    }
    delete checkRZ;
  }
  LogDebug("HitPairGeneratorFromLayerPair")<<" total number of pairs provided back: "<<result.size();
}
