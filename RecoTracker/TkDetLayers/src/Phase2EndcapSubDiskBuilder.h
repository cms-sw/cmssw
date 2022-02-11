#ifndef TkDetLayers_Phase2EndcapSubDiskBuilder_h
#define TkDetLayers_Phase2EndcapSubDiskBuilder_h

#include "Phase2EndcapSubDisk.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/** A concrete builder for Phase2EndcapSubDisk
 */

#pragma GCC visibility push(hidden)
class Phase2EndcapSubDiskBuilder {
public:
  Phase2EndcapSubDiskBuilder(){};
  Phase2EndcapSubDisk* build(const GeometricDet* aPhase2EndcapSubDisk, const TrackerGeometry* theGeomDetGeometry)
      __attribute__((cold));
};

#pragma GCC visibility pop
#endif
