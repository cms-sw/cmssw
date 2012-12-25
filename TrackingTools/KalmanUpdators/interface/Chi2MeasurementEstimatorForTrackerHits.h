#ifndef CommonDet_Chi2MeasurementEstimatorForTrackerHits_H
#define CommonDet_Chi2MeasurementEstimatorForTrackerHits_H

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"


/** A Chi2 Measurement Estimator. 
 *  Computhes the Chi^2 of a TrajectoryState with a RecHit or a 
 *  Plane. The TrajectoryState must have errors.
 *  Works for any RecHit dimension.
 */


class Chi2MeasurementEstimatorForTrackerHits : public Chi2MeasurementEstimatorBase {
public:

  /** Construct with cuts on chi2 and nSigma.
   *  The cut on Chi2 is used to define the acceptance of RecHits.
   *  The errors of the trajectory state are multiplied by nSigma 
   *  to define acceptance of Plane and maximalLocalDisplacement.
   */
  explicit Chi2MeasurementEstimatorForTrackerHits(double maxChi2, double nSigma = 3.) : 
    Chi2MeasurementEstimatorBase( maxChi2, nSigma), cacheUpToDate_(false) {}

  explicit Chi2MeasurementEstimatorForTrackerHits(const Chi2MeasurementEstimatorBase &est) :
        Chi2MeasurementEstimatorBase( est.chiSquaredCut(), est.nSigmaCut()), 
        cacheUpToDate_(false) {}
  
  void clearCache() { cacheUpToDate_ = false; }

  virtual std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				     const TransientTrackingRecHit&) const;

  virtual Chi2MeasurementEstimatorForTrackerHits* clone() const {
    return new Chi2MeasurementEstimatorForTrackerHits(*this);
  }
private:
        mutable bool cacheUpToDate_;
        mutable AlgebraicVector2     tsosMeasuredParameters_;
        mutable AlgebraicSymMatrix22 tsosMeasuredError_;
};

#endif
