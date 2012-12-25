#ifndef MRHChi2MeasurementEstimator_H
#define MRHChi2MeasurementEstimator_H

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

class SiTrackerMultiRecHitUpdator;

class MRHChi2MeasurementEstimator : public Chi2MeasurementEstimatorBase {
public:

  /** Construct with cuts on chi2 and nSigma.
   *  The cut on Chi2 is used to define the acceptance of RecHits.
   *  The errors of the trajectory state are multiplied by nSigma 
   *  to define acceptance of Plane and maximalLocalDisplacement.
   */

  explicit MRHChi2MeasurementEstimator(double maxChi2, double nSigma = 3.) : 
    Chi2MeasurementEstimatorBase( maxChi2, nSigma) {}

  virtual std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				     const TransientTrackingRecHit&) const;

  virtual MRHChi2MeasurementEstimator* clone() const {
    return new MRHChi2MeasurementEstimator(*this);
  }
  

};

#endif
