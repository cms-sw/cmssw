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
  
  GeometricSearchTracker* build(const GeometricDet* theGeometricTracker,
				const TrackerGeometry* theGeomDetGeometry);
};


#endif 
