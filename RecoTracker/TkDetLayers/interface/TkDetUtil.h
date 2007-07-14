#ifndef TkDetLayers_TkDetUtil_h
#define TkDetLayers_TkDetUtil_h

#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"


class BoundPlane;
class TrajectoryStateOnSurface;

namespace tkUtil {

float 
calculatePhiWindow( const MeasurementEstimator::Local2DVector& maxDistance, 
		    const TrajectoryStateOnSurface& ts, 
		    const BoundPlane& plane) const;


}

#endif // TkDetLayers_TkDetUtil_h
