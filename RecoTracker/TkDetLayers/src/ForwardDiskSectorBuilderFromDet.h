#ifndef RecoTracker_TkDetLayers_ForwardDiskSectorBuilderFromDet_h
#define RecoTracker_TkDetLayers_ForwardDiskSectorBuilderFromDet_h

#include "DataFormats/GeometrySurface/interface/BoundDiskSector.h"
#include "DataFormats/GeometrySurface/interface/DiskSectorBounds.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <utility>
#include <vector>
#include <iostream>

/** As it's name indicates, it's a builder of a BoundDiskSector from a collection of
 *  Dets. The disk sector has the minimal size fully containing all Dets.
 */

#pragma GCC visibility push(hidden)
class ForwardDiskSectorBuilderFromDet {
public:
  /// Warning, remember to assign this pointer to a ReferenceCountingPointer!
  /// Should be changed to return a ReferenceCountingPointer<BoundDisk>
  BoundDiskSector* operator()(const std::vector<const GeomDet*>& dets) const;

  std::pair<DiskSectorBounds*, GlobalVector> computeBounds(const std::vector<const GeomDet*>& dets) const;

private:
  Surface::RotationType computeRotation(const std::vector<const GeomDet*>& dets, Surface::PositionType pos) const;

  std::vector<GlobalPoint> computeTrapezoidalCorners(const GeomDet* detu) const;
};

#pragma GCC visibility pop
#endif
