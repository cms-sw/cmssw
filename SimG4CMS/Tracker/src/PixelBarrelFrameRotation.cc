#include "SimG4CMS/Tracker/interface/PixelBarrelFrameRotation.h"

Local3DPoint PixelBarrelFrameRotation::transformPoint(Local3DPoint & point,G4VPhysicalVolume * v=0) const {
  //
  // Please note PixelBarrel are NO MORE left handed since cms 132 - XML 1_2_5
  //
  // Please note PixelBarrel Detector are Left handed in DDD
  // G4 creates a PXBD_refl, which is right handed and there is where the hit is created
  // I write the hit in this coordinate frame (rh), since also ORCA generates a RH frame from the pixel
  // TkDDDInterface/interface/PlaneBuilder.h::makeRightHand
  //
  //  return Local3DPoint(point.x()/cm,-point.z()/cm,point.y()/cm);
  return Local3DPoint(point.x()/cm,point.z()/cm,-point.y()/cm);
}
