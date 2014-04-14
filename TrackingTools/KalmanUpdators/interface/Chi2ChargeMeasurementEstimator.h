#ifndef CommonDet_Chi2ChargeMeasurementEstimator_H
#define CommonDet_Chi2ChargeMeasurementEstimator_H

/** \class Chi2ChargeMeasurementEstimator
 *  A Chi2 Measurement Estimator. 
 *  Computhes the Chi^2 of a TrajectoryState with a RecHit or a 
 *  Plane. The TrajectoryState must have errors.
 *  Works for any RecHit dimension. Ported from ORCA.
 *
 *  \author todorov, cerati
 */

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
class SiStripCluster;
class TrackingRecHit;
class Chi2ChargeMeasurementEstimator GCC11_FINAL : public Chi2MeasurementEstimator {
public:

  /** Construct with cuts on chi2 and nSigma.
   *  The cut on Chi2 is used to define the acceptance of RecHits.
   *  The errors of the trajectory state are multiplied by nSigma 
   *  to define acceptance of Plane and maximalLocalDisplacement.
   */
  explicit Chi2ChargeMeasurementEstimator(double maxChi2, double nSigma,
	bool cutOnPixelCharge, bool cutOnStripCharge, double minGoodPixelCharge, double minGoodStripCharge) : 
    Chi2MeasurementEstimator( maxChi2, nSigma), cutOnPixelCharge_(cutOnPixelCharge),
    cutOnStripCharge_(cutOnStripCharge), minGoodPixelCharge_(minGoodPixelCharge),
    minGoodStripCharge_(minGoodStripCharge) {}

  using Chi2MeasurementEstimator::estimate;
  virtual std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				     const TrackingRecHit&) const;


  virtual Chi2ChargeMeasurementEstimator* clone() const {
    return new Chi2ChargeMeasurementEstimator(*this);
  }
private:
  bool cutOnPixelCharge_;
  bool cutOnStripCharge_;
  double minGoodPixelCharge_; 
  double minGoodStripCharge_;
  inline double minGoodCharge(int subdet) const {return (subdet>2?minGoodStripCharge_:minGoodPixelCharge_);}

  bool thickSensors (const SiStripDetId& detid) const;
  
  float sensorThickness (const DetId& detid) const;
  bool checkClusterCharge(const OmniClusterRef::ClusterStripRef cluster, float chargeCut) const;
  bool checkCharge(const TrackingRecHit& aRecHit, int subdet, float chargeCut) const;

};

#endif
