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

#include "OuterDetCompatibility.h"
#include "OuterHitCompatibility.h"

#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/Utilities/interface/Visibility.h"


template<typename Algo>
class dso_internal OuterEstimator final : public MeasurementEstimator {

public:

  using OuterHitCompat = OuterHitCompatibility<Algo>;

  OuterEstimator(
      const OuterDetCompatibility & detCompatibility,
      const OuterHitCompat & hitCompatibility,  
      const edm::EventSetup& iSetup) 
   : theDetCompatibility(detCompatibility), 
     theHitCompatibility (hitCompatibility) { }
  
  ~OuterEstimator() override{}

  std::pair<bool,double> estimate(
      const TrajectoryStateOnSurface& ts, 
      const TrackingRecHit& hit)  
    const override {
       return theHitCompatibility(hit) ? std::make_pair(true,1.) : std::make_pair(false,0.) ;
  }

  bool estimate(
      const TrajectoryStateOnSurface& ts, 
      const BoundPlane& plane
   ) const override {
    return theDetCompatibility(plane);
  }

  GlobalPoint center() { return theDetCompatibility.center(); }

  OuterEstimator* clone() const override {
    return new OuterEstimator(*this);
  }

  MeasurementEstimator::Local2DVector maximalLocalDisplacement( 
      const TrajectoryStateOnSurface& ts, const BoundPlane& plane) const override {
    return theDetCompatibility.maximalLocalDisplacement(
        ts.globalPosition(),plane);
 }

  const OuterDetCompatibility & detCompatibility() const 
    {return theDetCompatibility; }
  const OuterHitCompat & hitCompatibility() const 
    {return theHitCompatibility; }

private:
  OuterDetCompatibility theDetCompatibility;
  OuterHitCompat theHitCompatibility; 

};
#endif
