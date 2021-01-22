#ifndef RecoTracker_TkDetLayers_ForwardDiskSectorBuilderFromWedges_h
#define RecoTracker_TkDetLayers_ForwardDiskSectorBuilderFromWedges_h

#include "DataFormats/GeometrySurface/interface/BoundDiskSector.h"
#include "DataFormats/GeometrySurface/interface/DiskSectorBounds.h"
#include "TECWedge.h"
#include <utility>
#include <vector>

/** As it's name indicates, it's a builder of a BoundDiskSector from a collection of
 *  Wedges (of one petal). The disk sector has the minimal size fully containing all wedges.
 */

#pragma GCC visibility push(hidden)
class ForwardDiskSectorBuilderFromWedges {
public:
  /// Warning, remember to assign this pointer to a ReferenceCountingPointer!
  /// Should be changed to return a ReferenceCountingPointer<BoundDisk>
  BoundDiskSector* operator()(const std::vector<const TECWedge*>& wedges) const;

private:
  std::pair<DiskSectorBounds*, GlobalVector> computeBounds(const std::vector<const TECWedge*>& wedges) const;

  Surface::RotationType computeRotation(const std::vector<const TECWedge*>& wedges, Surface::PositionType pos) const;
};

#pragma GCC visibility pop
#endif
