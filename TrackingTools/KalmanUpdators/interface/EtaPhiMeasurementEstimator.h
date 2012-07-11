#ifndef CommonDet_EtaPhiMeasurementEstimator_H
#define CommonDet_EtaPhiMeasurementEstimator_H

/** \class EtaPhiMeasurementEstimator
 *  A EtaPhi Measurement Estimator. 
 *  Computhes the Chi^2 of a TrajectoryState with a RecHit or a 
 *  BoundPlane. The TrajectoryState must have errors.
 *  Works for any RecHit dimension. Ported from ORCA.
 *
 *  $Date: 2010/03/23 08:34:50 $
 *  $Revision: 1.4 $
 *  tschudi
 */

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

class EtaPhiMeasurementEstimator : public Chi2MeasurementEstimatorBase {
public:

  explicit EtaPhiMeasurementEstimator(double dEta, double dPhi) : 
    Chi2MeasurementEstimatorBase( 0.0, 0.0),
    thedEta(dEta),
    thedPhi(dPhi)
   {}
  ~EtaPhiMeasurementEstimator(){}

  std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				  const TransientTrackingRecHit&) const;

  virtual bool estimate(const TrajectoryStateOnSurface& tsos,
			const BoundPlane& plane) const;

  virtual Local2DVector maximalLocalDisplacement( const TrajectoryStateOnSurface& tsos,
						   const BoundPlane& plane) const;

  EtaPhiMeasurementEstimator* clone() const {
    return new EtaPhiMeasurementEstimator(*this);
  }
 private:
  double thedEta;
  double thedPhi;

};

#endif
