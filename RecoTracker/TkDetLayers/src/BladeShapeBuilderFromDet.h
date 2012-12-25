#ifndef RecoTracker_TkDetLayers_BladeShapeBuilderFromDet_h
#define RecoTracker_TkDetLayers_BladeShapeBuilderFromDet_h

#include "BoundDiskSector.h"
#include "DiskSectorBounds.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <utility>
#include <vector>
#include <iostream>


/** The trapezoid has the minimal size fully containing all Dets.
 */

#pragma GCC visibility push(hidden)
class BladeShapeBuilderFromDet {
 public:
  BoundDiskSector* operator()( const std::vector<const GeomDet*>& dets) const;
  

 private:
  std::pair<DiskSectorBounds*, GlobalVector>
  computeBounds( const std::vector<const GeomDet*>& dets,
		 const Plane& plane) const;

  Surface::RotationType
  computeRotation( const std::vector<const GeomDet*>& dets,
		   const Surface::PositionType& pos) const;

};

#pragma GCC visibility pop
#endif
