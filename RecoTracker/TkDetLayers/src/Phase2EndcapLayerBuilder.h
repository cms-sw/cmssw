#ifndef TkDetLayers_Phase2EndcapLayerBuilder_h
#define TkDetLayers_Phase2EndcapLayerBuilder_h

#include "Phase2EndcapLayer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2EndcapLayer
 */

#pragma GCC visibility push(hidden)
class Phase2EndcapLayerBuilder {
public:
  Phase2EndcapLayerBuilder(){};
  Phase2EndcapLayer* build(const GeometricDet* aPhase2EndcapLayer,
                           const TrackerGeometry* theGeomDetGeometry,
                           const bool useBrothers) __attribute__((cold));
};

#pragma GCC visibility pop
#endif
