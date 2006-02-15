#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossing2OrderLocal.h"
#include <algorithm>
#include <cmath>

HelixBarrelPlaneCrossing2OrderLocal::
HelixBarrelPlaneCrossing2OrderLocal( const GlobalPoint& startingPos,
				     const GlobalVector& startingDir,
				     double rho, const Plane& plane)
{
  // translate problem to local frame of the plane
  LocalPoint lPos = plane.toLocal(startingPos);
  LocalVector lDir = plane.toLocal(startingDir);

  // check if local frame is already special (local Y axis == global Z axis)
  LocalVector yPrime = plane.toLocal( GlobalVector(0,0,1.f));
  LocalVector diff = yPrime - LocalVector(0,-1.f,0);
  float sinPhi=0, cosPhi=0;
  bool rotated;
  Vector2D pos;
  Vector2D dir;

  if (diff.mag2() < 1.e-10) {

    // cout << "Plane already oriented, yPrime = " << yPrime << endl;

    // we are already in the special orientation
    pos = Vector2D( lPos.x(), lPos.y());
    dir = Vector2D( lDir.x(), lDir.y());;
    rotated = false;
  }
  else {

    // cout << "Plane needs rotation, yPrime = " << yPrime << endl;

    // we need to rotate the problem 
    sinPhi = yPrime.y();
    cosPhi = yPrime.x();
    pos = Vector2D( lPos.x()*cosPhi + lPos.y()*sinPhi,
		    -lPos.x()*sinPhi + lPos.y()*cosPhi);
    dir = Vector2D( lDir.x()*cosPhi + lDir.y()*sinPhi,
		    -lDir.x()*sinPhi + lDir.y()*cosPhi);
    rotated = true;
  }

  double d = -lPos.z();
  double x = pos.x() + dir.x()/lDir.z()*d - 0.5*rho*d*d;
  double y = pos.y() + dir.y()/lDir.z()*d;

//    cout << "d= " << d << ", pos.x()= " << pos.x() 
//         << ", dir.x()/lDir.z()= " << dir.x()/lDir.z() 
//         << ", 0.5*rho*d*d= " << 0.5*rho*d*d << endl;

  if (!rotated) {
    thePos = LocalPoint( x, y, 0);
    theDir = LocalVector( dir.x()+rho*d, dir.y(), lDir.z());
  }
  else {
    thePos = LocalPoint( x*cosPhi - y*sinPhi,
			 x*sinPhi + y*cosPhi, 0);
    float px = dir.x()+rho*d;
    theDir = LocalVector( px*cosPhi - dir.y()*sinPhi,
			  px*sinPhi + dir.y()*cosPhi, lDir.z());
  }
}

