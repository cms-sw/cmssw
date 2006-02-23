#ifndef TkDetLayers_GeometricSearchTrackerBuilder_h
#define TkDetLayers_GeometricSearchTrackerBuilder_h


#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

/** GeometricSearchTrackerBuilder implementation
 *  
 */

class GeometricSearchTrackerBuilder {
 public:

  GeometricSearchTrackerBuilder() {};
  ~GeometricSearchTrackerBuilder() {}; 
  
  GeometricSearchTracker* build(const GeometricDet* theGeometricTracker,
				const TrackingGeometry* theGeomDetGeometry);
};


#endif 
