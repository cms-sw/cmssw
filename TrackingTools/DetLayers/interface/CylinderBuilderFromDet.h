#ifndef DetLayers_CylinderBuilderFromDet_h
#define DetLayers_CylinderBuilderFromDet_h

/** \class CylinderBuilderFromDet
 *  Given a container of GeomDets, constructs a cylinder of minimal
 *  dimensions that contains all of the Dets completely (all corners
 *  etc.) 
 *  Useful for defining a BarrelDetLayer from a group of DetUnits.
 *
 *  $Date: 2007/08/22 16:07:44 $
 *  $Revision: 1.4 $
 */

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

#include <vector>
#include <limits>

class CylinderBuilderFromDet {
public:
  typedef GeomDet Det;
  typedef Surface::PositionType PositionType;
  typedef Surface::RotationType RotationType;
  typedef PositionType::BasicVectorType Vector;

  CylinderBuilderFromDet() : 
    rmin(std::numeric_limits<float>::max()), 
    rmax(0.0),
    zmin(std::numeric_limits<float>::max()),
    zmax(std::numeric_limits<float>::min()){}
  
  BoundCylinder* operator()( std::vector<const Det*>::const_iterator first,
			     std::vector<const Det*>::const_iterator last) const;

  void operator()(const Det& det);

  BoundCylinder* build() const;

private:
  float rmin;
  float rmax;
  float zmin;
  float zmax;

};

#endif
