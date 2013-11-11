#ifndef OuterEstimator_H
#define OuterEstimator_H

/** It is a MeasurementEstimator used by TrackingRegions for
    finding (OUTER) compatible hits and det units by testing the
    hit compatibility by OuterHitCompatibility and 
    det compatibility by OuterDetCompatibility */

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "RecoTracker/TkTrackingRegions/interface/OuterDetCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/OuterHitCompatibility.h"
#include "FWCore/Framework/interface/EventSetup.h"
//#include <utility>

class OuterEstimator : public MeasurementEstimator {

public:
  OuterEstimator(
      const OuterDetCompatibility & detCompatibility,
      const OuterHitCompatibility & hitCompatibility,  
      const edm::EventSetup& iSetup) 
   : theDetCompatibility(detCompatibility), 
     theHitCompatibility (hitCompatibility) { }
  virtual ~OuterEstimator(){}
  virtual std::pair<bool,double> estimate(
      const TrajectoryStateOnSurface& ts, 
      const TransientTrackingRecHit& hit)  
    const {
       return theHitCompatibility(&hit) ? std::make_pair(true,1.) : std::make_pair(false,0.) ;
  }

  virtual std::pair<bool,double> estimate(
      const TrajectoryStateOnSurface& ts, 
      const TrackingRecHit& hit,  
      const edm::EventSetup& iSetup) 
    const {
       return theHitCompatibility(&hit,iSetup) ? std::make_pair(true,1.) : std::make_pair(false,0.) ;
  }
 
  virtual bool estimate(
      const TrajectoryStateOnSurface& ts, 
      const BoundPlane& plane
) const {
    return theDetCompatibility(plane);
  }

  GlobalPoint center() { return theDetCompatibility.center(); }

  virtual OuterEstimator* clone() const {
    return new OuterEstimator(*this);
  }

  virtual MeasurementEstimator::Local2DVector maximalLocalDisplacement( 
      const TrajectoryStateOnSurface& ts, const BoundPlane& plane) const {
    return theDetCompatibility.maximalLocalDisplacement(
        ts.globalPosition(),plane);
 }

  const OuterDetCompatibility & detCompatibility() const 
    {return theDetCompatibility; }
  const OuterHitCompatibility & hitCompatibility() const 
    {return theHitCompatibility; }

private:
  OuterDetCompatibility theDetCompatibility;
  OuterHitCompatibility theHitCompatibility; 

};
#endif
