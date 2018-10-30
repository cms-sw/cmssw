#ifndef RecoTracker_TkTrackingRegions_OuterHitCompatibility_H
#define RecoTracker_TkTrackingRegions_OuterHitCompatibility_H


/** test compatibility of RecHit. 
    The phi and r-z are checked in independent way.
    The phi of a RecHit hit is tested if it is in the range 
    defined by OuterHitPhiPrediction.
    The r-z checking is done with a help of HitRZCompatibility checker */ 
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "OuterHitPhiPrediction.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/Math/interface/approx_atan2.h"

#include "FWCore/Utilities/interface/Visibility.h"

template<typename Algo>
class dso_internal OuterHitCompatibility {
public:

  OuterHitCompatibility(
      const OuterHitPhiPrediction & phiPrediction,
      const Algo  & rzCompatibility) 
    : thePhiPrediction(phiPrediction), 
      theRZCompatibility(rzCompatibility) {}

  bool operator() (const TrackingRecHit & hit) const {

     auto hitPos = hit.globalPosition();
     auto hitR = hitPos.perp();

     auto hitZ = hitPos.z();
     if ( !theRZCompatibility(hitR,hitZ) ) return false;

     auto hitPhi = unsafe_atan2f<9>(hitPos.y(),hitPos.x());

     return checkPhi(hitPhi, hitR);
   }


  bool checkPhi(float phi, float r) const {
    auto hitPhiRange = thePhiPrediction(r);
    bool phiOK = Geom::phiLess(hitPhiRange.min(),phi) && Geom::phiLess(phi,hitPhiRange.max());
    return phiOK;
  }

private:
  OuterHitPhiPrediction thePhiPrediction;
  Algo theRZCompatibility;
};
#endif
