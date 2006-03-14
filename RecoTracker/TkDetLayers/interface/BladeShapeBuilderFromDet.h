#ifndef RecoTracker_TkDetLayers_BladeShapeBuilderFromDet_h
#define RecoTracker_TkDetLayers_BladeShapeBuilderFromDet_h

#include "RecoTracker/TkDetLayers/interface/BoundDiskSector.h"
#include "RecoTracker/TkDetLayers/interface/DiskSectorBounds.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include <utility>
#include <vector>
#include <iostream>

using namespace std;

/** The trapezoid has the minimal size fully containing all Dets.
 */

class BladeShapeBuilderFromDet {
 public:
  BoundDiskSector* operator()( const vector<const GeomDet*>& dets) const;
  

 private:
  pair<DiskSectorBounds, GlobalVector>
  computeBounds( const vector<const GeomDet*>& dets,
		 const BoundPlane& plane) const;

  Surface::RotationType
  computeRotation( const vector<const GeomDet*>& dets,
		   const Surface::PositionType& pos) const;

};

#endif
