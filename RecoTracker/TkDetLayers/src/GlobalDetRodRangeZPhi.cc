#include "GlobalDetRodRangeZPhi.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

#include <array>

GlobalDetRodRangeZPhi::GlobalDetRodRangeZPhi( const Plane& plane) {

  float dx = plane.bounds().width()/2.;
  float dy = plane.bounds().length()/2.;
  float dz = plane.bounds().thickness()/2.;

  // rods may be inverted (actually are in every other layer), so have to find out the 
  // orientation of the local frame
  float deltaZ = (plane.toGlobal( LocalPoint( 0, 0, -dz)).perp() < 
                  plane.toGlobal( LocalPoint( 0, 0, dz)).perp() ) ? -dz : dz ;
  

  const std::array<Surface::GlobalPoint, 4> corners{{
      plane.toGlobal( LocalPoint( -dx, -dy, deltaZ)),
      plane.toGlobal( LocalPoint( -dx,  dy, deltaZ)),
      plane.toGlobal( LocalPoint(  dx, -dy, deltaZ)),
      plane.toGlobal( LocalPoint(  dx,  dy, deltaZ))}};

  float phimin = corners[0].phi();
  float phimax = phimin;

  float zmin   = corners[0].z();
  float zmax   = zmin;

  for ( const auto& corner : corners )
  {
    float phi = corner.phi();
    if ( Geom::phiLess( phi, phimin)) phimin = phi;
    if ( Geom::phiLess( phimax, phi)) phimax = phi;

    float z = corner.z();
    zmin = std::min(zmin, z);
    zmax = std::max(zmax, z);
  }

  theZRange.first    = zmin;
  theZRange.second   = zmax;
  thePhiRange.first  = phimin;
  thePhiRange.second = phimax;

}
