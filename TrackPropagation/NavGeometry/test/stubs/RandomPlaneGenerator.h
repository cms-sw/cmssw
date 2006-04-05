#ifndef RandomPlaneGenerator_H_
#define RandomPlaneGenerator_H_

#include "Geometry/Surface/interface/BoundPlane.h"
#include "Geometry/Surface/interface/ReferenceCounted.h"
#include "Geometry/Vector/interface/GlobalTag.h"
#include "Geometry/Vector/interface/Point3DBase.h"
#include "Geometry/Vector/interface/Vector3DBase.h"

/** \class RandomPlaneGenerator
 * Interface for classes generating random planes,
 * given a point and direction vector. */

class RandomPlaneGenerator {

public:
  typedef ReferenceCountingPointer<BoundPlane> PlanePtr;
  typedef Surface::GlobalPoint                 GlobalPoint;
  typedef Surface::GlobalVector                GlobalVector;

public:

  virtual ~RandomPlaneGenerator() {}

  /** Generate surface at a given point / direction.
   */
  virtual PlanePtr operator() (const GlobalPoint&,
			       const GlobalVector&) const=0;
  
protected:
  typedef Vector3DBase<double,GlobalTag> GlobalVectorDouble;
  typedef Point3DBase<double,GlobalTag> GlobalPointDouble;

};
#endif
