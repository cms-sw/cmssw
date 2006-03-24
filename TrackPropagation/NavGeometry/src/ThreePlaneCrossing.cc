#include "TrackPropagation/NavGeometry/src/ThreePlaneCrossing.h"
#include "TrackPropagation/NavGeometry/src/LinearEquation3.h"

Plane::GlobalPoint 
ThreePlaneCrossing::crossing( const Plane& a, const Plane& b, 
			      const Plane& c) const 
{
  typedef Plane::Scalar    T;

  const GlobalVector n1g( a.normalVector());
  const GlobalVector n2g( b.normalVector());
  const GlobalVector n3g( c.normalVector());

  Basic3DVector<T> n1(n1g.basicVector());
  Basic3DVector<T> n2(n2g.basicVector());
  Basic3DVector<T> n3(n3g.basicVector());
  Basic3DVector<T> rhs( n1.dot( a.position().basicVector()),
			n2.dot( b.position().basicVector()),
			n3.dot( c.position().basicVector()));
  LinearEquation3<T> solver;
  Plane::GlobalPoint result( solver.solution( n1, n2, n3, rhs));
  return result;
}
