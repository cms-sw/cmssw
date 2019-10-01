#ifndef TkDetLayers_Phase2OTBarrelLayerBuilder_h
#define TkDetLayers_Phase2OTBarrelLayerBuilder_h

#include "Phase2OTBarrelLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2OTBarrelLayer 
 */

#pragma GCC visibility push(hidden)
class Phase2OTBarrelLayerBuilder {
public:
  Phase2OTBarrelLayerBuilder(){};
  Phase2OTBarrelLayer* build(const GeometricDet* aPhase2OTBarrelLayer,
                             const TrackerGeometry* theGeomDetGeometry,
                             const bool useBrothers = true) __attribute__((cold));
};

#pragma GCC visibility pop
#endif
