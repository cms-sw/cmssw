#ifndef DetLayers_RodPlaneBuilderFromDet_H
#define DetLayers_RodPlaneBuilderFromDet_H

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "Geometry/Surface/interface/BoundPlane.h"
#include <utility>
#include <vector>

class RectangularPlaneBounds;

/** Builds the minimal rectangular box that contains all input Dets fully.
 */

class RodPlaneBuilderFromDet {
public:
  typedef GeomDet Det;
  
  /// Warning, remember to assign this pointer to a ReferenceCountingPointer!
  /// Should be changed to return a ReferenceCountingPointer<BoundPlane>
  BoundPlane* operator()( const vector<const Det*>& dets) const;

  pair<RectangularPlaneBounds, GlobalVector>
  computeBounds( const vector<const Det*>& dets, const BoundPlane& plane) const;

  Surface::RotationType 
  computeRotation( const vector<const Det*>& dets, 
		   const Surface::PositionType& meanPos) const; 

};

#endif
