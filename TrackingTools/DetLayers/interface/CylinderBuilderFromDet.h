#ifndef DetLayers_CylinderBuilderFromDet_h
#define DetLayers_CylinderBuilderFromDet_h

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <vector>
#include "Geometry/Surface/interface/BoundCylinder.h"

/** Given a container of GeomDets, constructs a cylinder of minimal
 *  dimensions that contains all of the Dets completely (all corners
 *  etc.) 
 *  Useful for defining a BarrelDetLayer from a group of DetUnits.
 */

class CylinderBuilderFromDet {
public:
  typedef GeomDet Det;
  typedef Surface::PositionType PositionType;
  typedef Surface::RotationType RotationType;

  BoundCylinder* operator()( std::vector<const Det*>::const_iterator first,
			     std::vector<const Det*>::const_iterator last) const;
};

#endif
