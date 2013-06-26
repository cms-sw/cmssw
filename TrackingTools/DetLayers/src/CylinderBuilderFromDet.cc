#include "TrackingTools/DetLayers/interface/CylinderBuilderFromDet.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/BoundingBox.h"
#include <algorithm>

using namespace std;

BoundCylinder* 
CylinderBuilderFromDet::operator()( vector<const Det*>::const_iterator first,
				    vector<const Det*>::const_iterator last) const
{
  // find mean position and radius
  typedef PositionType::BasicVectorType Vector;
  Vector posSum(0,0,0);
  float rSum = 0;
  for (vector<const Det*>::const_iterator i=first; i!=last; i++) {
    posSum += (**i).surface().position().basicVector();
    rSum += (**i).surface().position().perp();
  }
  float div(1/float(last-first));
  PositionType meanPos( div*posSum);
  float meanR( div*rSum);

  // find max deviations from mean pos in Z and from mean R
  float rmin = meanR;
  float rmax = meanR;
  float zmin = meanPos.z();
  float zmax = meanPos.z();
  for (vector<const Det*>::const_iterator i=first; i!=last; i++) {
    vector<GlobalPoint> corners = 
      BoundingBox::corners( dynamic_cast<const Plane&>((**i).surface()));
    for (vector<GlobalPoint>::const_iterator ic = corners.begin();
	 ic != corners.end(); ic++) {
      float r = ic->perp();
      float z = ic->z();
      rmin = min( rmin, r);
      rmax = max( rmax, r);
      zmin = min( zmin, z);
      zmax = max( zmax, z);
    }
    // in addition to the corners we have to check the middle of the 
    // det +/- thickness/2
    // , since the min  radius for some barrel dets is reached there
    float rdet = (**i).surface().position().perp();
    float halfThick = (**i).surface().bounds().thickness() / 2.F;
    rmin = min( rmin, rdet-halfThick);
    rmax = max( rmax, rdet+halfThick);
  }

  // the transverse position is zero by construction.
  // the Z position is the average between zmin and zmax, since the bounds 
  // are symmetric
  // for the same reason the R is the average between rmin and rmax,
  // but this is done by the Bounds anyway.

  PositionType pos( 0, 0, 0.5*(zmin+zmax));
  RotationType rot;      // only "barrel" orientation supported
  
  auto scp = new SimpleCylinderBounds( rmin, rmax, 
       				       zmin-pos.z(), zmax-pos.z());
  return new Cylinder(Cylinder::computeRadius(*scp), pos, rot, scp);

}

void CylinderBuilderFromDet::operator()(const Det& det) {
  BoundingBox bb( dynamic_cast<const Plane&>(det.surface()));
  for (int nc=0; nc<8; ++nc) {
    float r = bb[nc].perp();
    float z = bb[nc].z();
    rmin = std::min( rmin, r);
    rmax = std::max( rmax, r);
    zmin = std::min( zmin, z);
    zmax = std::max( zmax, z);
  }
  // in addition to the corners we have to check the middle of the 
  // det +/- thickness/2
  // , since the min  radius for some barrel dets is reached there
  float rdet = det.surface().position().perp();
  float halfThick = det.surface().bounds().thickness() / 2.F;
  rmin = std::min( rmin, rdet-halfThick);
  rmax = std::max( rmax, rdet+halfThick);
}

BoundCylinder* CylinderBuilderFromDet::build() const {
  
  PositionType pos( 0, 0, 0.5*(zmin+zmax));
  RotationType rot;      // only "barrel" orientation supported

  auto scp = new SimpleCylinderBounds( rmin, rmax,
                                       zmin-pos.z(), zmax-pos.z());
  return new Cylinder(Cylinder::computeRadius(*scp), pos, rot, scp);

}
