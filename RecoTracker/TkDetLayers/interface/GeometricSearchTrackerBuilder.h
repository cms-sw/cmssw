#ifndef TkDetLayers_GeometricSearchTrackerBuilder_h
#define TkDetLayers_GeometricSearchTrackerBuilder_h


#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

/** GeometricSearchTrackerBuilder implementation
 *  
 */

class GeometricSearchTrackerBuilder {
 public:

  GeometricSearchTrackerBuilder() {};
  ~GeometricSearchTrackerBuilder() {}; 
  
  GeometricSearchTracker* build(const GeometricDetPtr theGeometricTracker,
				const TrackerGeometry* theGeomDetGeometry);
};


#endif 
