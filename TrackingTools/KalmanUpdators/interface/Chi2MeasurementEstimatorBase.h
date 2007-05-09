#ifndef CommonDet_Chi2MeasurementEstimatorBase_H
#define CommonDet_Chi2MeasurementEstimatorBase_H

/** \class Chi2MeasurementEstimatorBase
 *  A base class for  Chi2 -- type of Measurement Estimators. 
 *  Implements common functionality. Ported from ORCA.
 *
 *  $Date: 2007/05/09 13:58:19 $
 *  $Revision: 1.1.2.1 $
 *  \author todorov, cerati
 */

#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

class Chi2MeasurementEstimatorBase : public MeasurementEstimator {
public:

  /** Construct with cuts on chi2 and nSigma.
   *  The cut on Chi2 is used to define the acceptance of RecHits.
   *  The errors of the trajectory state are multiplied by nSigma 
   *  to define acceptance of BoundPlane and maximalLocalDisplacement.
   */
  explicit Chi2MeasurementEstimatorBase(double maxChi2, double nSigma = 3.) : 
    theMaxChi2(maxChi2), theNSigma(nSigma) {}

  virtual bool estimate( const TrajectoryStateOnSurface& ts, 
			 const BoundPlane& plane) const;

  virtual Local2DVector 
  maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
			    const BoundPlane& plane) const;

  double chiSquaredCut() const {return theMaxChi2;}
  double nSigmaCut() const {return theNSigma;}

protected:

  std::pair<bool,double> returnIt( double est) const {
    return est > chiSquaredCut() ? HitReturnType(false,est) : HitReturnType(true,est);
  }

private:
  double theMaxChi2;
  double theNSigma;
};

#endif
