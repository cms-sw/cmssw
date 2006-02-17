#ifndef CD_Chi2Strip1DEstimator_H_
#define CD_Chi2Strip1DEstimator_H_

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

/** A Chi2 MeasurementEstimator that only uses the X coordinate
 *  in the measurement frame (the one perpendicular to the strip).
 */

class Chi2Strip1DEstimator : public Chi2MeasurementEstimatorBase {

public:

  explicit Chi2Strip1DEstimator(double maxChi2, double nSigma = 3.) : 
    Chi2MeasurementEstimatorBase( maxChi2, nSigma) {}

  virtual std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				     const TransientTrackingRecHit&) const;

  virtual Chi2Strip1DEstimator* clone() const {
    return new Chi2Strip1DEstimator(*this);
  }

};

#endif //CD_Chi2Strip1DEstimator_H_
