#ifndef CommonDet_Chi2MeasurementEstimator_H
#define CommonDet_Chi2MeasurementEstimator_H

/** \class Chi2MeasurementEstimator
 *  A Chi2 Measurement Estimator. 
 *  Computhes the Chi^2 of a TrajectoryState with a RecHit or a 
 *  Plane. The TrajectoryState must have errors.
 *  Works for any RecHit dimension. Ported from ORCA.
 *
 *  \author todorov, cerati
 */

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

class Chi2MeasurementEstimator : public Chi2MeasurementEstimatorBase {
public:

  using Chi2MeasurementEstimatorBase::Chi2MeasurementEstimatorBase;

  std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				     const TrackingRecHit&) const override;

  Chi2MeasurementEstimator* clone() const override {
    return new Chi2MeasurementEstimator(*this);
  }

};

#endif
