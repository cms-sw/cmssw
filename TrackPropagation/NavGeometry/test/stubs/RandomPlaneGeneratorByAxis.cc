#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include "TrackPropagation/NavGeometry/test/stubs/RandomPlaneGeneratorByAxis.h"
#include "CLHEP/Random/RandFlat.h"

#include <string>

using namespace std;

//
// Constructor
//
RandomPlaneGeneratorByAxis::RandomPlaneGeneratorByAxis () {
  // FIXME use ParameterSet
  theMaxTilt = 0.5;
  string planeTypeString = "arbitrary";
  theSize = 0.;


  if ( planeTypeString=="forward" )
    thePlaneType = forward;
  else if ( planeTypeString=="strictForward" )
    thePlaneType = strictForward;
  else if ( planeTypeString=="barrel" )
    thePlaneType = barrel;
  else if ( planeTypeString=="strictBarrel" )
    thePlaneType = strictBarrel;
  else
    thePlaneType = arbitrary;

}

RandomPlaneGeneratorByAxis::RandomPlaneGeneratorByAxis (const float maxTilt,
							const PlaneType type) :
  theMaxTilt(maxTilt),
  thePlaneType(type) {

  // FIXME use ParameterSet
  theSize = 0.;
}

//
// Generate random plane at position and with z-axis around direction.
//
RandomPlaneGenerator::PlanePtr
RandomPlaneGeneratorByAxis::operator() (const GlobalPoint& position,
					const GlobalVector& direction) const {
  //
  // convert to double
  //
  GlobalPointDouble doublePos(position.x(),position.y(),position.z());
  GlobalVectorDouble doubleDir(direction.x(),direction.y(),direction.z());
  //
  // reference direction (random sign w.r.t. input direction)
  //
  GlobalVectorDouble refDir(doubleDir.unit());
  if ( CLHEP::RandFlat::shoot(-1.,1.)<0. )  refDir *= -1;
  TkRotation<double> rotation;
  //
  // arbitrary or barrel planes
  //
  if ( thePlaneType==arbitrary || thePlaneType==barrel || thePlaneType==strictBarrel ) {
    //
    // for barrek planes use only transverse components of reference direction
    //
    if ( thePlaneType!=arbitrary )
      refDir = GlobalVectorDouble(refDir.x(),refDir.y(),0.).unit();
    //
    // now construct a plane with z-axis // reference direction, 
    // x-axis normal to z-local and z-global and 
    // y-axis according to a right-handed system
    //
    GlobalVectorDouble zFrame = refDir;
    GlobalVectorDouble xFrame;
    if ( fabs(zFrame.z())<0.99 ) {
      xFrame = zFrame.cross(GlobalVectorDouble(0.,0.,1.)).unit();
    }
    else {
      xFrame = GlobalVectorDouble(0.,1.,0.).cross(zFrame).unit();
    }
    GlobalVector yFrame = zFrame.cross(xFrame);
    rotation = TkRotation<double>(xFrame.x(),xFrame.y(),xFrame.z(),
				  yFrame.x(),yFrame.y(),yFrame.z(),
				  zFrame.x(),zFrame.y(),zFrame.z());
    //
    // Random rotation around z-local to determine direction 
    // of tilt (restrict to 0 or pi in case of barrel plane
    // to keep orientation for strictBarrel planes).
    //
    double dPhi = CLHEP::RandFlat::shoot(0.,2*M_PI);
    if ( thePlaneType!=arbitrary )
      dPhi = dPhi>M_PI ? M_PI : 0.;
    rotation = rotationAroundZ(dPhi)*rotation;
    //
    // now introduce tilt by a random rotation around
    // y-local (to keep y-local || z-global for the
    // case of strictBarrel planes)
    //
    double cosTilt = CLHEP::RandFlat::shoot(cos(theMaxTilt),1.);
    rotation = rotationAroundY(cosTilt<=1.?acos(cosTilt):0.)*rotation;
  }
  //
  // forward planes
  //
  else {
    //
    // Start with strictForward plane with sign(z) according 
    // to the reference direction
    //
    GlobalVectorDouble zFrame(0.,0.,refDir.z()>0.?1.:-1.);
    GlobalVectorDouble xFrame = GlobalVectorDouble(0.,1.,0.).cross(zFrame);
    GlobalVectorDouble yFrame = zFrame.cross(xFrame);
    rotation = TkRotation<double>(xFrame.x(),xFrame.y(),xFrame.z(),
				  yFrame.x(),yFrame.y(),yFrame.z(),
				  zFrame.x(),zFrame.y(),zFrame.z());
  }
  //
  // Add random rotation around z-local (to randomize azimuth),
  // unless it's a "strict" barrel or forward plane
  //
  if ( thePlaneType!=strictBarrel && thePlaneType!=strictForward )
    rotation = rotationAroundZ(CLHEP::RandFlat::shoot(0.,2*M_PI))*rotation;
  //
  // Now define the origin of the plane by a random shift
  // w.r.t. the reference position
  //
  Basic3DVector<double> deltaPos = 
    rotation.multiplyInverse(Basic3DVector<double>(CLHEP::RandFlat::shoot(-theSize/2,theSize/2),
						   CLHEP::RandFlat::shoot(-theSize/2,theSize/2),
						   0));
  doublePos += GlobalVectorDouble(deltaPos);
  //
  // Construct and return plane
  //
//    cout << "Reference direction was " << doubleDir.unit() << endl;
//    cout << "Rotation is " <<
//      rotation.xx()*(rotation.yy()*rotation.zz()-rotation.yz()*rotation.zy()) -
//      rotation.yy()*(rotation.zz()*rotation.xx()-rotation.zx()*rotation.xz()) +
//      rotation.zz()*(rotation.xx()*rotation.yy()-rotation.xy()*rotation.yx());
//    cout << rotation << endl;
#ifndef CMS_NO_COMPLEX_RETURNS
  RandomPlaneGenerator::PlanePtr plane = 
    PlaneBuilder().plane(GlobalPoint(doublePos.x(),doublePos.y(),doublePos.z()),
			 TkRotation<float>(rotation.xx(),rotation.xy(),rotation.xz(),
					   rotation.yx(),rotation.yy(),rotation.yz(),
					   rotation.zx(),rotation.zy(),rotation.zz()));
  return plane;
#else
  return PlaneBuilder().plane(GlobalPoint(doublePos.x(),doublePos.y(),doublePos.z()),
			      TkRotation<float>(rotation.xx(),rotation.xy(),rotation.xz(),
						rotation.yx(),rotation.yy(),rotation.yz(),
						rotation.zx(),rotation.zy(),rotation.zz()));
#endif
}
//
// rotation around x-axis by angle tilt
//
TkRotation<double>
RandomPlaneGeneratorByAxis::rotationAroundX(const double tilt) const {
  return TkRotation<double>(1.,0.,0.,
			    0.,cos(tilt),sin(tilt),
			    0.,-sin(tilt),cos(tilt));
}
//
// rotation around y-axis by angle tilt
//
TkRotation<double>
RandomPlaneGeneratorByAxis::rotationAroundY(const double tilt) const {
  return TkRotation<double>(cos(tilt),0.,sin(tilt),
			    0.,1.,0.,
			    -sin(tilt),0.,cos(tilt));
}
//
// rotation around z-axis by angle tilt
//
TkRotation<double>
RandomPlaneGeneratorByAxis::rotationAroundZ(const double tilt) const {
  return TkRotation<double>(cos(tilt),sin(tilt),0.,
			    -sin(tilt),cos(tilt),0.,
			    0.,0.,1.);
}

