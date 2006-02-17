#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
//#include "CommonReco/PatternTools/interface/MeasurementExtractor.h"
//#include "CommonDet/BasicDet/interface/RecHit.h"
//#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//#include "CommonDet/DetGeometry/interface/BoundPlane.h"
#include "Geometry/Surface/interface/BoundPlane.h"

bool Chi2MeasurementEstimatorBase::estimate( const TrajectoryStateOnSurface& ts, 
					     const BoundPlane& plane) const
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
							const BoundPlane& plane) const
{
  if ( ts.hasError()) {
    LocalError le = ts.localError().positionError();
    return Local2DVector( sqrt(le.xx())*nSigmaCut(), sqrt(le.yy())*nSigmaCut());
  }
  else return Local2DVector(0,0);
}
