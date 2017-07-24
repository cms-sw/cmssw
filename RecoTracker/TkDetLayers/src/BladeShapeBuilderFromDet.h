#ifndef RecoTracker_TkDetLayers_BladeShapeBuilderFromDet_h
#define RecoTracker_TkDetLayers_BladeShapeBuilderFromDet_h

#include "BoundDiskSector.h"
#include "DiskSectorBounds.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <utility>
#include <vector>
#include <iostream>


/** The trapezoid has the minimal size fully containing all Dets.
 */

#pragma GCC visibility push(hidden)
class BladeShapeBuilderFromDet {
 public:
  static BoundDiskSector * build( const std::vector<const GeomDet*>& dets) __attribute__ ((cold));
};

#pragma GCC visibility pop
#endif
