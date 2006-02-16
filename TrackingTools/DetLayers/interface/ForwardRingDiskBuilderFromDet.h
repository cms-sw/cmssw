#ifndef DetLayers_ForwardRingDiskBuilderFromDet_H
#define DetLayers_ForwardRingDiskBuilderFromDet_H

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "Geometry/Surface/interface/BoundDisk.h"
#include <utility>
#include <vector>

class Det;
class SimpleDiskBounds;

/** As it's name indicates, it's a builder of BoundDisk from a collection of
 *  Dets. The disk has the minimal size fully containing all Dets.
 */

class ForwardRingDiskBuilderFromDet {
public:

  /// Warning, remember to assign this pointer to a ReferenceCountingPointer!
  /// Should be changed to return a ReferenceCountingPointer<BoundDisk>
  BoundDisk* operator()( const vector<const GeomDet*>& dets) const;
  
  pair<SimpleDiskBounds, float>
  computeBounds( const vector<const GeomDet*>& dets) const;

};

#endif
