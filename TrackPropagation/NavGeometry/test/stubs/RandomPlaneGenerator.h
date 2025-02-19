#ifndef RandomPlaneGenerator_H_
#define RandomPlaneGenerator_H_

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"

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
