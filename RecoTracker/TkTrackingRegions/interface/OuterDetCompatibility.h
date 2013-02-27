#ifndef OuterDetCompatibility_H
#define OuterDetCompatibility_H

/** check det compatibility by comparistion of det BoundPlane ranges
    with phi,r,z ranges (given at construction). */

#include "RecoTracker/TkTrackingRegions/interface/OuterHitPhiPrediction.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZConstraint.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"

class OuterDetCompatibility  {
public:

  OuterDetCompatibility(const BarrelDetLayer* layer,
      const OuterHitPhiPrediction::Range & phiRange,
      const HitRZConstraint::Range & rRange,
      const HitRZConstraint::Range & zRange)
    : theLayer(layer), barrel(true), 
      hitDetPhiRange(phiRange), hitDetRRange(rRange), hitDetZRange(zRange) { } 

  OuterDetCompatibility(const ForwardDetLayer* layer,
      const OuterHitPhiPrediction::Range & phiRange,
      const HitRZConstraint::Range & rRange,
      const HitRZConstraint::Range & zRange)
    : theLayer(layer), barrel(false), 
      hitDetPhiRange(phiRange), hitDetRRange(rRange), hitDetZRange(zRange) { } 

   bool operator() (const BoundPlane& plane) const;

   MeasurementEstimator::Local2DVector maximalLocalDisplacement( 
       const TrajectoryStateOnSurface& ts, const BoundPlane& plane) const;
   MeasurementEstimator::Local2DVector maximalLocalDisplacement(
       const GlobalPoint & ts, const BoundPlane& plane) const;

   GlobalPoint center() const;

   const OuterHitPhiPrediction::Range & phiRange()const {return hitDetPhiRange;}
   const HitRZConstraint::Range &rRange() const { return hitDetRRange; }
   const HitRZConstraint::Range &zRange() const { return hitDetZRange; }

private:
   bool checkPhi(const OuterHitPhiPrediction::Range & detPhiRange) const;
   bool checkR(const HitRZConstraint::Range & detRRange) const;
   bool checkZ(const HitRZConstraint::Range & detZRange) const;

   double loc_dist(
      double radius, double ts_phi, double range_phi, double cosGamma) const;

private:
    const DetLayer* theLayer;
    bool barrel;
    OuterHitPhiPrediction::Range hitDetPhiRange;
    HitRZConstraint::Range hitDetRRange, hitDetZRange;
};

#endif
