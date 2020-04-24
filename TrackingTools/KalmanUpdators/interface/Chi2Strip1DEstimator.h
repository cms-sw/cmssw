#ifndef CD_Chi2Strip1DEstimator_H_
#define CD_Chi2Strip1DEstimator_H_

/** \class Chi2Strip1DEstimator
 *  A Chi2 MeasurementEstimator that only uses the X coordinate
 *  in the measurement frame (the one perpendicular to the strip).
 *  Ported from ORCA.
 *
 *  \author todorov, cerati
 */

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

class Chi2Strip1DEstimator final : public Chi2MeasurementEstimatorBase {

public:

  explicit Chi2Strip1DEstimator(double maxChi2, double nSigma = 3.) : 
    Chi2MeasurementEstimatorBase( maxChi2, nSigma) {}

  std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				     const TrackingRecHit&) const override;

  Chi2Strip1DEstimator* clone() const override {
    return new Chi2Strip1DEstimator(*this);
  }

};

#endif //CD_Chi2Strip1DEstimator_H_
