#ifndef CommonDet_Chi2ChargeMeasurementEstimator_H
#define CommonDet_Chi2ChargeMeasurementEstimator_H

/** \class Chi2ChargeMeasurementEstimator
 *  A Chi2 Measurement Estimator, checking also the charge of the cluster.
 *  Computhes the Chi^2 of a TrajectoryState with a RecHit or a 
 *  Plane. The TrajectoryState must have errors.
 *  If the cluster passes the chi2 cut, the charge is checked. This cut can 
 *  be bypassed for high-pt cut.
 *  Works for any RecHit dimension.
 *
 *  \author todorov, cerati, speer
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
	bool cutOnPixelCharge, bool cutOnStripCharge, double minGoodPixelCharge, double minGoodStripCharge,
	float pTChargeCutThreshold = 100000.) : 
    Chi2MeasurementEstimator( maxChi2, nSigma), cutOnPixelCharge_(cutOnPixelCharge),
    cutOnStripCharge_(cutOnStripCharge), minGoodPixelCharge_(minGoodPixelCharge),
    minGoodStripCharge_(minGoodStripCharge) {
      if (pTChargeCutThreshold>=0.) pTChargeCutThreshold_=pTChargeCutThreshold;
      else pTChargeCutThreshold_=100000;
    }

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
  float pTChargeCutThreshold_;
  inline double minGoodCharge(int subdet) const {return (subdet>2?minGoodStripCharge_:minGoodPixelCharge_);}

  bool thickSensors (const SiStripDetId& detid) const;
  
  float sensorThickness (const DetId& detid) const;
  bool checkClusterCharge(const OmniClusterRef::ClusterStripRef cluster, float chargeCut) const;
  bool checkCharge(const TrackingRecHit& aRecHit, int subdet, float chargeCut) const;

};

#endif
