#ifndef Chi2SwitchingEstimator_H_
#define Chi2SwitchingEstimator_H_

/** \class Chi2SwitchingEstimator
 *  A measurement estimator that uses Chi2MeasurementEstimator for
 *  pixel and matched strip hits, and Chi2StripEstimator for
 *  simple strip hits. Ported from ORCA.
 *
 *  $Date: 2007/05/09 13:50:25 $
 *  $Revision: 1.4 $
 *  \author todorov, cerati
 */

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2StripEstimator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"

class Chi2SwitchingEstimator : public Chi2MeasurementEstimatorBase {

public:

  explicit Chi2SwitchingEstimator(double aMaxChi2, double nSigma = 3.) : 
    Chi2MeasurementEstimatorBase(aMaxChi2,nSigma),
    theLocalEstimator(new Chi2MeasurementEstimator(aMaxChi2,nSigma)),
    theStripEstimator(new Chi2StripEstimator(aMaxChi2,nSigma)) {}

  /// implementation of MeasurementEstimator::estimate
  virtual std::pair<bool,double> estimate (const TrajectoryStateOnSurface& aTsos,
				      const TransientTrackingRecHit& aHit) const;

  virtual Chi2SwitchingEstimator* clone() const 
  {
    return new Chi2SwitchingEstimator(*this);
  }

private:
  /// estimator for 2D hits (matched or pixel)
  const Chi2MeasurementEstimator& localEstimator() const {
    return *theLocalEstimator;
  }
  /// estimator for 1D hits (non-matched strips)
  const Chi2StripEstimator& stripEstimator() const {
    return *theStripEstimator;
  }

private:
  DeepCopyPointerByClone<const Chi2MeasurementEstimator> theLocalEstimator;
  DeepCopyPointerByClone<const Chi2StripEstimator>       theStripEstimator;

};
#endif //Chi2SwitchingEstimator_H_
