#ifndef RandomCylinderGenerator_H_
#define RandomCylinderGenerator_H_

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
//#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"

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
