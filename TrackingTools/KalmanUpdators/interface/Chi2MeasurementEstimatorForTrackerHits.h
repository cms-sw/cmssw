#ifndef CommonDet_Chi2MeasurementEstimatorForTrackerHits_H
#define CommonDet_Chi2MeasurementEstimatorForTrackerHits_H

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include <atomic>

/** A Chi2 Measurement Estimator. 
 *  Computhes the Chi^2 of a TrajectoryState with a RecHit or a 
 *  Plane. The TrajectoryState must have errors.
 *  Works for any RecHit dimension.
 */


class AlgebraicHelper {
    public:
        AlgebraicHelper(const AlgebraicVector2& v, const AlgebraicSymMatrix22& m) :
            tsosMeasuredParameters_(v), tsosMeasuredError_(m) {}
        const AlgebraicVector2 params() const {return tsosMeasuredParameters_;}
        const AlgebraicSymMatrix22 errors() const {return tsosMeasuredError_;}
    private:
        AlgebraicVector2     tsosMeasuredParameters_;
        AlgebraicSymMatrix22 tsosMeasuredError_;
};
class Chi2MeasurementEstimatorForTrackerHits GCC11_FINAL : public Chi2MeasurementEstimatorBase {
public:

  Chi2MeasurementEstimatorForTrackerHits(const Chi2MeasurementEstimatorForTrackerHits& src);
  /** Construct with cuts on chi2 and nSigma.
   *  The cut on Chi2 is used to define the acceptance of RecHits.
   *  The errors of the trajectory state are multiplied by nSigma 
   *  to define acceptance of Plane and maximalLocalDisplacement.
   */
  explicit Chi2MeasurementEstimatorForTrackerHits(double maxChi2, double nSigma = 3.) : 
    Chi2MeasurementEstimatorBase( maxChi2, nSigma), aHelper(nullptr) {}

  explicit Chi2MeasurementEstimatorForTrackerHits(const Chi2MeasurementEstimatorBase &est) :
        Chi2MeasurementEstimatorBase( est.chiSquaredCut(), est.nSigmaCut()), 
        aHelper(nullptr) {}
  
  void clearCache() { aHelper = nullptr; }

  virtual std::pair<bool,double> estimate(const TrajectoryStateOnSurface&,
				     const TransientTrackingRecHit&) const;

  virtual Chi2MeasurementEstimatorForTrackerHits* clone() const {
    return new Chi2MeasurementEstimatorForTrackerHits(*this);
  }
private:
        mutable std::atomic<AlgebraicHelper*> aHelper;
};

#endif
