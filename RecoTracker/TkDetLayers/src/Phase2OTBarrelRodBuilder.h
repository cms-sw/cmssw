#ifndef TkDetLayers_Phase2OTBarrelRodBuilder_h
#define TkDetLayers_Phase2OTBarrelRodBuilder_h

#include "Phase2OTBarrelRod.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2OTBarrelRod 
 */

#pragma GCC visibility push(hidden)
class Phase2OTBarrelRodBuilder {
public:
  Phase2OTBarrelRodBuilder(){};
  Phase2OTBarrelRod* build(const GeometricDet* thePhase2OTBarrelRod,
                           const TrackerGeometry* theGeomDetGeometry,
                           const bool useBrothers = true) __attribute__((cold));
};

#pragma GCC visibility pop
#endif
