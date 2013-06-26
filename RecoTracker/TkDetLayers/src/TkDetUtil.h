#ifndef TkDetLayers_TkDetUtil_h
#define TkDetLayers_TkDetUtil_h

#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#pragma GCC visibility push(hidden)

class GeomDet;
class Plane;
class TrajectoryStateOnSurface;

namespace tkDetUtil {

  bool overlapInPhi( const GlobalPoint& crossPoint,const GeomDet & det, float phiWindow);

  float computeWindowSize( const GeomDet* det, 
				   const TrajectoryStateOnSurface& tsos, 
				   const MeasurementEstimator& est);


  float 
  calculatePhiWindow( const MeasurementEstimator::Local2DVector& maxDistance, 
		      const TrajectoryStateOnSurface& ts, 
		      const Plane& plane);


}

#pragma GCC visibility pop
#endif // TkDetLayers_TkDetUtil_h
