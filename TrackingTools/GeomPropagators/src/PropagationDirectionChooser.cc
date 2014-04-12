#include "TrackingTools/GeomPropagators/interface/PropagationDirectionChooser.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include <cmath>

PropagationDirection
PropagationDirectionChooser::operator() (const FreeTrajectoryState& fts, 
					 const Surface& surface) const
{
  // understand the surface type (expect cylinder/plane)
  // unfortunately dynamic_cast on Sun is broken -- cannot deal with const
  // so here we get rid of const
  const Surface* sur = (const Surface*)&surface;
  const Cylinder* bc = dynamic_cast<const Cylinder*>(sur);
  const Plane* bp = dynamic_cast<const Plane*>(sur);
  if (bc != 0) {
    //
    // cylinder
    //
    return (*this)(fts, *bc);
  }
  else if (bp != 0) {
    //
    // plane
    //
    return (*this)(fts, *bp);
  }
  else {
    throw PropagationException("The surface is neither Cylinder nor Plane");
  }
}

PropagationDirection
PropagationDirectionChooser::operator() (const FreeTrajectoryState& fts, 
					 const Plane& plane) const 
{
  // propagation towards arbitrary plane
  // need computation of intersection point between plane 
  // and track (straight line approximation) to set direction
  // Copied from BidirectionalPropagator.
  PropagationDirection dir = anyDirection;

  GlobalPoint x = fts.position();
  GlobalVector p = fts.momentum().unit();
  GlobalPoint sp = plane.toGlobal(LocalPoint(0.,0.));
  GlobalVector v = plane.toGlobal(LocalVector(1.,0.,0.));
  GlobalVector w = plane.toGlobal(LocalVector(0.,1.,0.));
  AlgebraicMatrix33 a;
  a(0,0) = v.x(); a(0,1) = w.x(); a(0,2) = -p.x();
  a(1,0) = v.y(); a(1,1) = w.y(); a(1,2) = -p.y();
  a(2,0) = v.z(); a(2,1) = w.z(); a(2,2) = -p.z();
  AlgebraicVector3 b;
  b[0] = x.x() - sp.x();
  b[1] = x.y() - sp.y();
  b[2] = x.z() - sp.z();

  int ifail = !a.Invert();
  if (ifail == 0) {
    // plane and momentum are not parallel
    b = a*b;
    // b[2] nb of steps along unit vector parallel to momentum to reach plane

    const double small = 1.e-4; // 1 micron distance
    if (fabs(b[2]) < small) { 
      // already on plane, choose forward as default
      dir = alongMomentum;
    } 
    else if (b[2] < 0.) {
      dir = oppositeToMomentum;
    }
    else {
      dir = alongMomentum;    
    }
  } 
  return dir;
}


PropagationDirection
PropagationDirectionChooser::operator() (const FreeTrajectoryState& fts, 
					 const Cylinder& cylinder) const
{
  // propagation to cylinder with axis along z
  // Copied from BidirectionalPropagator.
  PropagationDirection dir = anyDirection;

  GlobalPoint sp = cylinder.toGlobal(LocalPoint(0.,0.));
  if (sp.x() != 0. || sp.y() != 0.) {
    throw PropagationException("Cannot propagate to an arbitrary cylinder");
  }

  GlobalPoint x = fts.position();
  GlobalVector p = fts.momentum();

  const double small = 1.e-4; // 1 micron distance
  double rdiff = x.perp() - cylinder.radius();
  
  if ( fabs(rdiff) < small ) { 
    // already on cylinder, choose forward as default
    dir = alongMomentum;
  } 
  else {
    int rSign = ( rdiff >= 0. ) ? 1 : -1 ;
    if ( rSign == -1 ) {
      // starting point inside cylinder 
      // propagate always in direction of momentum
      dir = alongMomentum;
    }
    else {
      // starting point outside cylinder
      // choose direction so as to approach cylinder surface
      double proj = (x.x()*p.x() + x.y()*p.y()) * rSign;
      if (proj < 0.) {
	dir = alongMomentum;
      }
      else {
	dir = oppositeToMomentum;    
      }
    }
  }
  return dir;
}
