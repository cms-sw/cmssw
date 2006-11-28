#ifndef RandomCylinderGenerator_H_
#define RandomCylinderGenerator_H_

#include "Geometry/Surface/interface/BoundCylinder.h"
//#include "Geometry/Surface/interface/ReferenceCounted.h"
#include "Geometry/Surface/interface/SimpleCylinderBounds.h"
#include "Geometry/Vector/interface/GlobalTag.h"
#include "Geometry/Vector/interface/Point3DBase.h"
#include "Geometry/Vector/interface/Vector3DBase.h"

/** \class RandomCylinderGenerator
 * Generates a (not yet random) cylinder at a given position.
 */

class RandomCylinderGenerator {

public:
  //  typedef ReferenceCountingPointer<BoundCylinder> CylinderPtr;

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
  BoundCylinder::BoundCylinderPointer operator() (const GlobalPoint& position,
			  const GlobalVector& direction) const 
  {
    const GlobalPointDouble pos(position.x(),position.y(),position.z());
    const GlobalPointDouble pos0(0,0,0);
    
#ifndef CMS_NO_COMPLEX_RETURNS

      BoundCylinder::BoundCylinderPointer cylinder =
      BoundCylinder::build(pos0,
			   TkRotation<float>(),
			   pos.perp(),
			   SimpleCylinderBounds(pos.perp(),pos.perp(),-theMaxZ,theMaxZ));
    return cylinder;
#else
    return BoundCylinder::build(pos0,
				TkRotation<float>(),
				pos.perp(),
				SimpleCylinderBounds(pos.perp(),pos.perp(),
						     -theMaxZ,theMaxZ));
#endif
  }

private:
  float theMaxZ;
};
#endif
