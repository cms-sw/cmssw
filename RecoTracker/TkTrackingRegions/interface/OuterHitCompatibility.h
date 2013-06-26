#ifndef RecoTracker_TkTrackingRegions_OuterHitCompatibility_H
#define RecoTracker_TkTrackingRegions_OuterHitCompatibility_H


/** test compatibility of RecHit. 
    The phi and r-z are checked in independent way.
    The phi of a RecHit hit is tested if it is in the range 
    defined by OuterHitPhiPrediction.
    The r-z checking is done with a help of HitRZCompatibility checker */ 
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkTrackingRegions/interface/OuterHitPhiPrediction.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
class OuterHitCompatibility {
public:

  OuterHitCompatibility(
      const OuterHitPhiPrediction & phiPrediction,
      const HitRZCompatibility & rzCompatibility) 
    : thePhiPrediction(phiPrediction) 
  { theRZCompatibility = rzCompatibility.clone(); }

  OuterHitCompatibility(const OuterHitCompatibility & ohc) 
    : thePhiPrediction(ohc.thePhiPrediction)
  { theRZCompatibility = ohc.theRZCompatibility->clone(); } 

   ~OuterHitCompatibility() 
   { delete theRZCompatibility; }  


  bool operator() (const TransientTrackingRecHit * hit) const;
  bool operator() (const TrackingRecHit* hit,  const edm::EventSetup& iSetup) const;

  bool checkPhi(const float & phi, const float & r) const;

  OuterHitCompatibility* clone() const { 
    return new OuterHitCompatibility(*this);
  }

protected:
  const HitRZCompatibility * theRZCompatibility;
  OuterHitPhiPrediction thePhiPrediction;
};
#endif
