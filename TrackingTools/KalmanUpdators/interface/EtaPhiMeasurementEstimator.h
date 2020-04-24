#ifndef CommonDet_EtaPhiMeasurementEstimator_H
#define CommonDet_EtaPhiMeasurementEstimator_H

/** \class EtaPhiMeasurementEstimator
 *  A EtaPhi Measurement Estimator. 
 *  Computhes the Chi^2 of a TrajectoryState with a RecHit or a 
 *  Plane. The TrajectoryState must have errors.
 *  Works for any RecHit dimension. Ported from ORCA.
 *
 *  tschudi
 */

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"

class EtaPhiMeasurementEstimator final : public MeasurementEstimator {
public:

  explicit EtaPhiMeasurementEstimator(float dEta, float dPhi) : 
    thedEta(dEta),
    thedPhi(dPhi)
   {}
  ~EtaPhiMeasurementEstimator() override{}

  std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				  const TrackingRecHit&) const override;

  bool estimate(const TrajectoryStateOnSurface& tsos,
			const Plane& plane) const override;

  Local2DVector maximalLocalDisplacement( const TrajectoryStateOnSurface& tsos,
						   const Plane& plane) const override;

  EtaPhiMeasurementEstimator* clone() const override {
    return new EtaPhiMeasurementEstimator(*this);
  }
 private:
  float thedEta;
  float thedPhi;

};

#endif
