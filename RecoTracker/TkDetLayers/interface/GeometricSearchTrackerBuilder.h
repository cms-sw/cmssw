#ifndef TkDetLayers_GeometricSearchTrackerBuilder_h
#define TkDetLayers_GeometricSearchTrackerBuilder_h

#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class TrackerTopology;
class MTDTopology;

/** GeometricSearchTrackerBuilder implementation
 *  
 */

class GeometricSearchTrackerBuilder {
public:
  GeometricSearchTrackerBuilder() {}
  ~GeometricSearchTrackerBuilder() {}

  GeometricSearchTracker* build(const GeometricDet* theGeometricTracker,
                                const TrackerGeometry* theGeomDetGeometry,
                                const TrackerTopology* tTopo,
                                const bool usePhase2Stacks = false) __attribute__((cold));

  //This constructor builds also the MTD geometry
  GeometricSearchTracker* build(const GeometricDet* theGeometricTracker,
                                const TrackerGeometry* theGeomDetGeometry,
                                const TrackerTopology* tTopo,
                                const MTDGeometry* mtd,
                                const MTDTopology* mTopo,
                                const bool usePhase2Stacks = false) __attribute__((cold));
};

#endif
