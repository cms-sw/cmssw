#ifndef ConvertFromToCLHEP_H
#define ConvertFromToCLHEP_H

//CLHEP
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Vector/ThreeVector.h"

//CMS
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

/** Utilities to convert among CLHEP and CMS points and vectors
 */

namespace TrackPropagation {
  /** Convert a CMS GlobalPoint to a CLHEP HepPoint3D
   */
  static HepPoint3D globalPointToPoint3D(const GlobalPoint& p) {
    return HepPoint3D(p.x(), p.y(), p.z());
  }

  /** Convert a CLHEP HepPoint3D to a CMS GlobalPoint 
   */
  static GlobalPoint point3DToGlobalPoint(const HepPoint3D& p) {
    return GlobalPoint(p.x(), p.y(), p.z());
  }


  /** Convert a CMS GlobalVector to a CLHEP HepNormal3D
   */
  static HepNormal3D globalVectorToNormal3D(const GlobalVector& p) {
    return HepNormal3D(p.x(), p.y(), p.z());
  }

  /** Convert a CLHEP HepNormal3D to a CMS GlobalVector 
   */
  static GlobalVector normal3DToGlobalVector(const HepNormal3D& p) {
    return GlobalVector(p.x(), p.y(), p.z());
  }




  /** Convert a CMS GlobalVector to a CLHEP Hep3Vector
   */
  static Hep3Vector globalVectorTo3Vector(const GlobalVector& p) {
    return Hep3Vector(p.x(), p.y(), p.z());
  }

  /** Convert a CLHEP Hep3Vector to a CMS GlobalVector 
   */
  static GlobalVector hep3VectorToGlobalVector(const Hep3Vector& p) {
    return GlobalVector(p.x(), p.y(), p.z());
  }




  /** Convert a CMS GlobalPoint to a CLHEP Hep3Vector
   */
  static Hep3Vector globalVectorTo3Vector(const GlobalPoint& p) {
    return Hep3Vector(p.x(), p.y(), p.z());
  }

  /** Convert a CLHEP Hep3Vector to a CMS GlobalPoint 
   */
  static GlobalPoint hep3VectorToGlobalPoint(const Hep3Vector& p) {
    return GlobalPoint(p.x(), p.y(), p.z());
  }
}



#endif
