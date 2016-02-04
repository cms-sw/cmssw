#ifndef CD_Chi2StripEstimator_H_
#define CD_Chi2StripEstimator_H_

/** \class Chi2StripEstimator
 *  A Chi2 MeasurementEstimator that works in the measurement (strip) frame
 *  and uses both coordinates of a hit. Ported from ORCA.
 *
 *  $Date: 2007/05/09 14:05:13 $
 *  $Revision: 1.2 $
 *  \author todorov, cerati
 */

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"

class Chi2StripEstimator : public Chi2MeasurementEstimatorBase {
public:

  explicit Chi2StripEstimator(double maxChi2, double nSigma = 3.) : 
    Chi2MeasurementEstimatorBase( maxChi2, nSigma) {}

  virtual std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				     const TransientTrackingRecHit&) const;
  
  virtual Chi2StripEstimator* clone() const {
    return new Chi2StripEstimator(*this);
  }

};

#endif //CD_Chi2StripEstimator_H_
