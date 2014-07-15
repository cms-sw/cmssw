#ifndef GsfChi2MeasurementEstimator_H
#define GsfChi2MeasurementEstimator_H

// #include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include <utility>

/** Class which calculates the chisquare of a predicted Gaussian mixture
 *  trajectory state with respect to a
 *  reconstructed hit according to the Gaussian-sum filter (GSF) strategy.
 *  The relevant formulas can be found in
 *  R. Fruhwirth, Computer Physics Communications 100 (1997), 1.
 */

class GsfChi2MeasurementEstimator : public Chi2MeasurementEstimatorBase {
public:

  GsfChi2MeasurementEstimator() : 
    Chi2MeasurementEstimatorBase(100.),
    theEstimator(100.) {}

  GsfChi2MeasurementEstimator(double aMaxChi2) : 
    Chi2MeasurementEstimatorBase(aMaxChi2),
    theEstimator(aMaxChi2) {}

  virtual std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
					  const TrackingRecHit&) const;

  virtual GsfChi2MeasurementEstimator* clone() const {
    return new GsfChi2MeasurementEstimator(*this);
  }

private:
  Chi2MeasurementEstimator theEstimator;
};

#endif
