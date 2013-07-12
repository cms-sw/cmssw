#ifndef DetLayers_ForwardRingDiskBuilderFromDet_H
#define DetLayers_ForwardRingDiskBuilderFromDet_H

/** \class ForwardRingDiskBuilderFromDet
 *  As it's name indicates, it's a builder of BoundDisk from a collection of
 *  Dets. The disk has the minimal size fully containing all Dets.
 *
 *  $Date: 2007/03/07 16:28:39 $
 *  $Revision: 1.3 $
 */

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

#include <utility>
#include <vector>

class Det;
class SimpleDiskBounds;

class ForwardRingDiskBuilderFromDet {
public:

  /// Warning, remember to assign this pointer to a ReferenceCountingPointer!
  /// Should be changed to return a ReferenceCountingPointer<BoundDisk>
  BoundDisk* operator()( const std::vector<const GeomDet*>& dets) const;
  
  std::pair<SimpleDiskBounds *, float>
  computeBounds( const std::vector<const GeomDet*>& dets) const;

};

#endif
