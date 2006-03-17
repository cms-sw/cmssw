#include "TrackingTools/DetLayers/interface/CylinderBuilderFromDet.h"
//#include "CommonDet/DetGeometry/interface/BoundPlane.h"
#include "Geometry/Surface/interface/SimpleCylinderBounds.h"
#include "Geometry/Surface/interface/BoundingBox.h"
#include <algorithm>

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
      BoundingBox().corners( dynamic_cast<const BoundPlane&>((**i).surface()));
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
  
  return new BoundCylinder( pos, rot, 
			    SimpleCylinderBounds( rmin, rmax, 
						  zmin-pos.z(), zmax-pos.z()));
}
