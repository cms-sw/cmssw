#ifndef RandomCylinderGenerator_H_
#define RandomCylinderGenerator_H_

#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/ReferenceCounted.h"
#include "Geometry/Surface/interface/CylinderBuilder.h"
#include "Geometry/Surface/interface/SimpleCylinderBounds.h"
#include "Geometry/Vector/interface/GlobalTag.h"
#include "Geometry/Vector/interface/Point3DBase.h"
#include "Geometry/Vector/interface/Vector3DBase.h"

/** \class RandomCylinderGenerator
 * Generates a (not yet random) cylinder at a given position.
 */

class RandomCylinderGenerator {

public:
  typedef ReferenceCountingPointer<BoundCylinder> CylinderPtr;

private:
  typedef Vector3DBase<double,GlobalTag> GlobalVectorDouble;
  typedef Point3DBase<double,GlobalTag> GlobalPointDouble;

public:
  //
  // Constructor/Destructor
  //
  RandomCylinderGenerator(const float maxZ) :
    theMaxZ(maxZ) {}
  ~RandomCylinderGenerator() {}
  /** Generate cylinder at a given point.
   */
  CylinderPtr operator() (const GlobalPoint& position,
			  const GlobalVector& direction) const 
  {
    GlobalPointDouble pos(position.x(),position.y(),position.z());
#ifndef CMS_NO_COMPLEX_RETURNS
    CylinderPtr cylinder =
      CylinderBuilder().cylinder(GlobalPoint(0.,0.,0.),
				 TkRotation<float>(),
				 SimpleCylinderBounds(pos.perp(),pos.perp(),
						      -theMaxZ,theMaxZ));
    return cylinder;
#else
    return CylinderBuilder().cylinder(GlobalPoint(0.,0.,0.),
				      TkRotation<float>(),
				      SimpleCylinderBounds(pos.perp(),pos.perp(),
							   -theMaxZ,theMaxZ));
#endif
  }

private:
  float theMaxZ;
};
#endif
