#ifndef RecoTracker_TkDetLayers_ForwardDiskSectorBuilderFromDet_h
#define RecoTracker_TkDetLayers_ForwardDiskSectorBuilderFromDet_h

#include "RecoTracker/TkDetLayers/interface/BoundDiskSector.h"
#include "RecoTracker/TkDetLayers/interface/DiskSectorBounds.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include <utility>
#include <vector>
#include <iostream>

using namespace std;

/** As it's name indicates, it's a builder of a BoundDiskSector from a collection of
 *  Dets. The disk sector has the minimal size fully containing all Dets.
 */

class ForwardDiskSectorBuilderFromDet {
public:

  /// Warning, remember to assign this pointer to a ReferenceCountingPointer!
  /// Should be changed to return a ReferenceCountingPointer<BoundDisk>
  BoundDiskSector* operator()( const vector<const GeomDet*>& dets) const;
  
  pair<DiskSectorBounds, GlobalVector>
  computeBounds( const vector<const GeomDet*>& dets) const;

private:

  Surface::RotationType
  computeRotation( const vector<const GeomDet*>& dets, Surface::PositionType pos) const;

  vector<GlobalPoint> 
  computeTrapezoidalCorners( const GeomDet* detu) const;

};

#endif
