#ifndef OuterDetCompatibility_H
#define OuterDetCompatibility_H

/** check det compatibility by comparistion of det BoundPlane ranges
    with phi,r,z ranges (given at construction). */

#include "RecoTracker/TkTrackingRegions/interface/OuterHitPhiPrediction.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZConstraint.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"

class OuterDetCompatibility  {
public:
  typedef PixelRecoRange<float> Range;

  OuterDetCompatibility(const BarrelDetLayer* layer,
      const OuterHitPhiPrediction::Range & phiRange,
      const Range & rRange,
      const Range & zRange)
    : theLayer(layer), barrel(true), 
      hitDetPhiRange(phiRange), hitDetRRange(rRange), hitDetZRange(zRange) { } 

  OuterDetCompatibility(const ForwardDetLayer* layer,
      const OuterHitPhiPrediction::Range & phiRange,
      const Range & rRange,
      const Range & zRange)
    : theLayer(layer), barrel(false), 
      hitDetPhiRange(phiRange), hitDetRRange(rRange), hitDetZRange(zRange) { } 

   bool operator() (const BoundPlane& plane) const;

   MeasurementEstimator::Local2DVector maximalLocalDisplacement( 
       const TrajectoryStateOnSurface& ts, const BoundPlane& plane) const;
   MeasurementEstimator::Local2DVector maximalLocalDisplacement(
       const GlobalPoint & ts, const BoundPlane& plane) const;

   GlobalPoint center() const;

   const OuterHitPhiPrediction::Range & phiRange()const {return hitDetPhiRange;}
   const Range & rRange() const { return hitDetRRange; }
   const Range & zRange() const { return hitDetZRange; }

private:
   bool checkPhi(const OuterHitPhiPrediction::Range & detPhiRange) const;
   bool checkR(const Range & detRRange) const;
   bool checkZ(const Range & detZRange) const;

   double loc_dist(
      double radius, double ts_phi, double range_phi, double cosGamma) const;

private:
  const DetLayer* theLayer;
  bool barrel;
  OuterHitPhiPrediction::Range hitDetPhiRange;
  Range hitDetRRange, hitDetZRange;
};

#endif
