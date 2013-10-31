#ifndef TkDetLayers_GeometricSearchTrackerBuilder_h
#define TkDetLayers_GeometricSearchTrackerBuilder_h


#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class TrackerTopology;

/** GeometricSearchTrackerBuilder implementation
 *  
 */

class GeometricSearchTrackerBuilder {
 public:

  GeometricSearchTrackerBuilder() {};
  ~GeometricSearchTrackerBuilder() {}; 
  
  GeometricSearchTracker* build(const GeometricDet* theGeometricTracker,
				const TrackerGeometry* theGeomDetGeometry,
				const TrackerTopology* tTopo);
};


#endif 
