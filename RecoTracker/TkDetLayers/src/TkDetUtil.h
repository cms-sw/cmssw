#ifndef TkDetLayers_TkDetUtil_h
#define TkDetLayers_TkDetUtil_h

#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

class GeomDet;
class BoundPlane;
class TrajectoryStateOnSurface;

namespace tkDetUtil {

  bool overlapInPhi( const GlobalPoint& crossPoint,const GeomDet & det, float phiWindow);

  float computeWindowSize( const GeomDet* det, 
				   const TrajectoryStateOnSurface& tsos, 
				   const MeasurementEstimator& est);


  float 
  calculatePhiWindow( const MeasurementEstimator::Local2DVector& maxDistance, 
		      const TrajectoryStateOnSurface& ts, 
		      const BoundPlane& plane);


}

#endif // TkDetLayers_TkDetUtil_h
