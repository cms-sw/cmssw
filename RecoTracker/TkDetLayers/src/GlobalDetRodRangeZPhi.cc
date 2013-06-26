#include "GlobalDetRodRangeZPhi.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"

#include <vector>

using namespace std;

GlobalDetRodRangeZPhi::GlobalDetRodRangeZPhi( const Plane& plane) {

  float dx = plane.bounds().width()/2.;
  float dy = plane.bounds().length()/2.;
  float dz = plane.bounds().thickness()/2.;

  // rods may be inverted (actually are in every other layer), so have to find out the 
  // orientation of the local frame
  float deltaZ = (plane.toGlobal( LocalPoint( 0, 0, -dz)).perp() < 
		  plane.toGlobal( LocalPoint( 0, 0, dz)).perp() ) ? -dz : dz ;
  

  vector<Surface::GlobalPoint> corners(4);
  corners[0] = plane.toGlobal( LocalPoint( -dx, -dy, deltaZ));
  corners[1] = plane.toGlobal( LocalPoint( -dx,  dy, deltaZ));
  corners[2] = plane.toGlobal( LocalPoint(  dx, -dy, deltaZ));
  corners[3] = plane.toGlobal( LocalPoint(  dx,  dy, deltaZ));

  float phimin = corners[0].phi();  float phimax = phimin;
  float zmin   = corners[0].z();    float zmax   = zmin;
  for ( int i=1; i<4; i++) {
    float phi = corners[i].phi();
    if ( PhiLess()( phi, phimin)) phimin = phi;
    if ( PhiLess()( phimax, phi)) phimax = phi;

    float z = corners[i].z();
    if ( z < zmin) zmin = z;
    if ( z > zmax) zmax = z;
  }

  theZRange.first    = zmin;
  theZRange.second   = zmax;
  thePhiRange.first  = phimin;
  thePhiRange.second = phimax;

}
