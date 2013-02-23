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

namespace {
  template<class T> inline T sqr( T t) {return t*t;}
}


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

  std::cout << "layers " << theInnerLayer.detLayer()->seqNum()  << " " << theOuterLayer.detLayer()->seqNum() << std::endl;

  constexpr float nSigmaRZ = std::sqrt(12.f);
  constexpr float nSigmaPhi = 3.f;
  for (int io = 0; io!=int(outerHitsMap.theHits.size()); ++io) { 
    Hit const & ohit =  outerHitsMap.theHits[io].hit();
    PixelRecoRange<float> phiRange = deltaPhi(outerHitsMap.r[io], 
					      outerHitsMap.phi[io], 
					      outerHitsMap.z[io], 
					      nSigmaPhi*outerHitsMap.drphi[io]
					      );    

    if (phiRange.empty()) continue;

    const HitRZCompatibility *checkRZ = region.checkRZ(theInnerLayer.detLayer(), ohit, iSetup,theOuterLayer.detLayer());
    if(!checkRZ) continue;

    auto innerRange = innerHitsMap.doubleRange(phiRange.min(), phiRange.max());
    LogDebug("HitPairGeneratorFromLayerPair")<<
      "preparing for combination of: "<< innerRange[1]-innerRange[0]+innerRange[3]-innerRange[2]
				      <<" inner and: "<< outerHitsMap.theHits.size()<<" outter";
    for(int j=0; j<3; j+=2)
      for (int i=innerRange[j]; i!=innerRange[j+1];++i) {  
      Range allowed = checkRZ->range(innerHitsMap.u[i]);
      float vErr = nSigmaRZ * innerHitsMap.dv[i];
      Range hitRZ(innerHitsMap.v[i]-vErr, innerHitsMap.v[i]+vErr);
      Range crossRange = allowed.intersection(hitRZ);
      if (! crossRange.empty() ) {
	if (theMaxElement!=0 && result.size() >= theMaxElement){
	  result.clear();
	  edm::LogError("TooManyPairs")<<"number of pairs exceed maximum, no pairs produced";
	  delete checkRZ;
	  return;
	}
        result.push_back( OrderedHitPair( innerHitsMap.theHits[i].hit(), ohit) );
      }
    }
    delete checkRZ;
  }
  LogDebug("HitPairGeneratorFromLayerPair")<<" total number of pairs provided back: "<<result.size();
}
