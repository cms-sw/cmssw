#ifndef CommonDet_Chi2MeasurementEstimatorBase_H
#define CommonDet_Chi2MeasurementEstimatorBase_H

/** \class Chi2MeasurementEstimatorBase
 *  A base class for  Chi2 -- type of Measurement Estimators. 
 *  Implements common functionality. Ported from ORCA.
 *
 *  \author todorov, cerati
 */

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include<limits>

class Chi2MeasurementEstimatorBase : public MeasurementEstimator {
public:

  /** Construct with cuts on chi2 and nSigma.
   *  The cut on Chi2 is used to define the acceptance of RecHits.
   *  The errors of the trajectory state are multiplied by nSigma 
   *  to define acceptance of Plane and maximalLocalDisplacement.
   */
  explicit Chi2MeasurementEstimatorBase(double maxChi2, double nSigma = 3., float maxDisp=std::numeric_limits<float>::max()) : 
    theMaxChi2(maxChi2), theNSigma(nSigma), theMaxDisplacement(maxDisp) {}

  template<typename... Args>
  Chi2MeasurementEstimatorBase(double maxChi2, double nSigma, float maxDisp,
                               Args && ...args) :
    MeasurementEstimator(args...),
    theMaxChi2(maxChi2), theNSigma(nSigma), theMaxDisplacement(maxDisp)  {}


  virtual std::pair<bool, double> estimate(const TrajectoryStateOnSurface& ts,
					   const TrackingRecHit &) const = 0;

  virtual bool estimate( const TrajectoryStateOnSurface& ts, 
			 const Plane& plane) const final;

  virtual Local2DVector 
  maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
			    const Plane& plane) const final;

  double chiSquaredCut() const {return theMaxChi2;}
  double nSigmaCut() const {return theNSigma;}

protected:

  std::pair<bool,double> returnIt( double est) const {
    return est > chiSquaredCut() ? HitReturnType(false,est) : HitReturnType(true,est);
  }

private:
  const double theMaxChi2;
  const double theNSigma;
  const float  theMaxDisplacement;
};

#endif
