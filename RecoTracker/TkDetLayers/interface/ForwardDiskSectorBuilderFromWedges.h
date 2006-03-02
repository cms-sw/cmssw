#ifndef RecoTracker_TkDetLayers_ForwardDiskSectorBuilderFromWedges_h
#define RecoTracker_TkDetLayers_ForwardDiskSectorBuilderFromWedges_h

#include "RecoTracker/TkDetLayers/interface/BoundDiskSector.h"
#include "RecoTracker/TkDetLayers/interface/DiskSectorBounds.h"
#include "RecoTracker/TkDetLayers/interface/TECWedge.h"
#include <utility>
#include <vector>

using namespace std;

/** As it's name indicates, it's a builder of a BoundDiskSector from a collection of
 *  Wedges (of one petal). The disk sector has the minimal size fully containing all wedges.
 */

class ForwardDiskSectorBuilderFromWedges {
public:

  /// Warning, remember to assign this pointer to a ReferenceCountingPointer!
  /// Should be changed to return a ReferenceCountingPointer<BoundDisk>
  BoundDiskSector* operator()( const vector<const TECWedge*>& wedges) const;

private:  
  pair<DiskSectorBounds, GlobalVector>
  computeBounds( const vector<const TECWedge*>& wedges) const;

  Surface::RotationType
  computeRotation( const vector<const TECWedge*>& wedges, Surface::PositionType pos) const;

};
 
#endif
