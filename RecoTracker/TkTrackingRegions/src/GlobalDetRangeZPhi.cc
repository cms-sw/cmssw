#include "RecoTracker/TkTrackingRegions/interface/GlobalDetRangeZPhi.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include <functional>
#include <cmath>
#include <vector>

using namespace std;

GlobalDetRangeZPhi::GlobalDetRangeZPhi( const BoundPlane& plane) {

  float dx = plane.bounds().width()/2.;
  float dy = plane.bounds().length()/2.;

  vector<Surface::GlobalPoint> corners(4);
  corners[0] = plane.toGlobal( LocalPoint( -dx, -dy, 0));
  corners[1] = plane.toGlobal( LocalPoint( -dx,  dy, 0));
  corners[2] = plane.toGlobal( LocalPoint(  dx, -dy, 0));
  corners[3] = plane.toGlobal( LocalPoint(  dx,  dy, 0));

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
