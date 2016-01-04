#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

bool Chi2MeasurementEstimatorBase::estimate( const TrajectoryStateOnSurface& ts, 
					     const Plane& plane) const
{
  if ( ts.hasError()) {
    return plane.bounds().inside( ts.localPosition(), 
				  ts.localError().positionError(),
				  nSigmaCut());
  }
  else return plane.bounds().inside(ts.localPosition());
}

MeasurementEstimator::Local2DVector 
Chi2MeasurementEstimatorBase::maximalLocalDisplacement( const TrajectoryStateOnSurface& ts,
							const Plane& plane) const
{
  constexpr float emax=100.;   // 0.5; 
  if ( ts.hasError()) {
    LocalError le = ts.localError().positionError();
    return Local2DVector( std::min(emax,std::sqrt(float(le.xx())))*nSigmaCut(), std::min(emax,std::sqrt(float(le.yy())))*nSigmaCut());
  }
  else return Local2DVector(0,0);
}
