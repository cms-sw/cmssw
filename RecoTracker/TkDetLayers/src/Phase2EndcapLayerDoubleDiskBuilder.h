#ifndef TkDetLayerDoubleDisks_Phase2EndcapLayerDoubleDiskBuilder_h
#define TkDetLayerDoubleDisks_Phase2EndcapLayerDoubleDiskBuilder_h

#include "Phase2EndcapLayerDoubleDisk.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2EndcapLayerDoubleDisk
 */

#pragma GCC visibility push(hidden)
class Phase2EndcapLayerDoubleDiskBuilder {
public:
  Phase2EndcapLayerDoubleDiskBuilder(){};
  Phase2EndcapLayerDoubleDisk* build(const GeometricDet* aPhase2EndcapLayerDoubleDisk,
                                     const TrackerGeometry* theGeomDetGeometry) __attribute__((cold));
};

#pragma GCC visibility pop
#endif
