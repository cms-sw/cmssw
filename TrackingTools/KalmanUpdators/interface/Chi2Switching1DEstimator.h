#ifndef Chi2Switching1DEstimator_H_
#define Chi2Switching1DEstimator_H_

/** \class Chi2Switching1DEstimator
 *  A measurement estimator that uses Chi2MeasurementEstimator for
 *  pixel and matched strip hits, and Chi2Strip1DEstimator for
 *  simple strip hits. Ported from ORCA.
 *
 *  $Date: 2013/01/10 12:12:18 $
 *  $Revision: 1.6 $
 *  \author todorov, cerati
 */

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2Strip1DEstimator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

class Chi2Switching1DEstimator GCC11_FINAL : public Chi2MeasurementEstimatorBase {

public:

  explicit Chi2Switching1DEstimator(double aMaxChi2, double nSigma = 3.) : 
    Chi2MeasurementEstimatorBase(aMaxChi2,nSigma),
    theLocalEstimator(aMaxChi2,nSigma),
    theStripEstimator(aMaxChi2,nSigma) {}

  /// implementation of MeasurementEstimator::estimate
  virtual std::pair<bool, double> estimate(const TrajectoryStateOnSurface& aTsos,
				      const TransientTrackingRecHit& aHit) const;

  virtual Chi2Switching1DEstimator* clone() const 
  {
    return new Chi2Switching1DEstimator(*this);
  }

private:
  /// estimator for 2D hits (matched or pixel)
  const Chi2MeasurementEstimator& localEstimator() const {
    return theLocalEstimator;
  }
  /// estimator for 1D hits (non-matched strips)
  const Chi2Strip1DEstimator& stripEstimator() const {
    return theStripEstimator;
  }

private:
  const Chi2MeasurementEstimator theLocalEstimator;
  const Chi2Strip1DEstimator     theStripEstimator;

};
#endif //Chi2Switching1DEstimator_H_



