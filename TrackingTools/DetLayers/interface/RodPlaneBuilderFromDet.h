#ifndef DetLayers_RodPlaneBuilderFromDet_H
#define DetLayers_RodPlaneBuilderFromDet_H

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include <utility>
#include <vector>

class RectangularPlaneBounds;

/** Builds the minimal rectangular box that contains all input Dets fully.
 */

class RodPlaneBuilderFromDet {
public:
  typedef GeomDet Det;
  
  /// Warning, remember to assign this pointer to a ReferenceCountingPointer!
  /// Should be changed to return a ReferenceCountingPointer<Plane>
  Plane* operator()( const std::vector<const Det*>& dets) const;

  std::pair<RectangularPlaneBounds, GlobalVector>
  computeBounds( const std::vector<const Det*>& dets, const Plane& plane) const;

  Surface::RotationType 
  computeRotation( const std::vector<const Det*>& dets, 
		   const Surface::PositionType& meanPos) const; 

};

#endif
