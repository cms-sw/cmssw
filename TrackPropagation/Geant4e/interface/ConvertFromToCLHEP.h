#ifndef TrackPropagation_ConvertFromToCLHEP_h
#define TrackPropagation_ConvertFromToCLHEP_h

//CLHEP
#include "CLHEP/Geometry/Point3D.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Units/SystemOfUnits.h"

//CMS
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Surface/interface/TkRotation.h"

/** Utilities to convert among CLHEP and CMS points and vectors
 */

namespace TrackPropagation {
  /** Convert a CMS GlobalPoint to a CLHEP HepPoint3D
      CMS uses cm while Geant4 uses mm. This is taken into account in the
      conversion.
   */
  static HepPoint3D globalPointToHepPoint3D(const GlobalPoint& r) {
    return HepPoint3D(r.x()*cm, r.y()*cm, r.z()*cm);
  }

  /** Convert a CLHEP HepPoint3D to a CMS GlobalPoint 
      CMS uses cms while Geant4 uses mm. This is taken into account in the
      conversion.
   */
  GlobalPoint hepPoint3DToGlobalPoint(const HepPoint3D& r) {
    return GlobalPoint(r.x()/cm, r.y()/cm, r.z()/cm);
  }


  /** Convert a CMS GlobalVector to a CLHEP HepNormal3D
   */
  HepNormal3D globalVectorToHepNormal3D(const GlobalVector& p) {
    return HepNormal3D(p.x(), p.y(), p.z());
  }

  /** Convert a CLHEP HepNormal3D to a CMS GlobalVector 
   */
  GlobalVector hepNormal3DToGlobalVector(const HepNormal3D& p) {
    return GlobalVector(p.x(), p.y(), p.z());
  }




  /** Convert a CMS GlobalVector to a CLHEP Hep3Vector
   */
  Hep3Vector globalVectorToHep3Vector(const GlobalVector& p) {
    return Hep3Vector(p.x(), p.y(), p.z());
  }

  /** Convert a CLHEP Hep3Vector to a CMS GlobalVector 
   */
  GlobalVector hep3VectorToGlobalVector(const Hep3Vector& p) {
    return GlobalVector(p.x(), p.y(), p.z());
  }




  /** Convert a CMS GlobalPoint to a CLHEP Hep3Vector
      CMS uses cm while Geant4 uses mm. This is taken into account in the
      conversion.
   */
  Hep3Vector globalPointToHep3Vector(const GlobalPoint& r) {
    return Hep3Vector(r.x()*cm, r.y()*cm, r.z()*cm);
  }

  /** Convert a CLHEP Hep3Vector to a CMS GlobalPoint 
      CMS uses cm while Geant4 uses mm. This is taken into account in the
      conversion.
   */
  GlobalPoint hep3VectorToGlobalPoint(const Hep3Vector& v) {
    return GlobalPoint(v.x()/cm, v.y()/cm, v.z()/cm);
  }









  /** Convert a CMS TkRotation<float> to a CLHEP HepRotation=G4RotationMatrix
   */
  HepRotation tkRotationFToHepRotation(const TkRotation<float>& tkr) {
    return HepRotation(Hep3Vector(tkr.xx(),tkr.yx(), tkr.zx()),
		       Hep3Vector(tkr.xy(),tkr.yy(), tkr.zy()),
		       Hep3Vector(tkr.xz(),tkr.yz(), tkr.zz()));
  }

  /** Convert a CLHEP Hep3Vector to a CMS GlobalPoint 
   */
  TkRotation<float> hepRotationToTkRotationF(const HepRotation& r) {
    return TkRotation<float>(r.xx(), r.xy(), r.xz(),
			     r.yx(), r.yy(), r.yz(),
			     r.zx(), r.zy(), r.zz());
  }
}



#endif
