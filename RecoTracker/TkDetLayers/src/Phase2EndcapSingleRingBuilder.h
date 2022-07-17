#ifndef TkDetLayers_Phase2EndcapSingleRingBuilder_h
#define TkDetLayers_Phase2EndcapSingleRingBuilder_h

#include "Phase2EndcapSingleRing.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2EndcapSingleRing 
 */

#pragma GCC visibility push(hidden)
class Phase2EndcapSingleRingBuilder {
public:
  Phase2EndcapSingleRingBuilder(){};
  Phase2EndcapSingleRing* build(const GeometricDet* aPhase2EndcapSingleRing, const TrackerGeometry* theGeomDetGeometry)
      __attribute__((cold));
};

#pragma GCC visibility pop
#endif
