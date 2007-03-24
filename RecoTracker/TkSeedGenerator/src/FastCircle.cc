#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"

FastCircle::FastCircle(const GlobalPoint& outerHit,
		       const GlobalPoint& middleHit,
		       const GlobalPoint& aVertex) : 
  theOuterPoint(outerHit), 
  theInnerPoint(middleHit), 
  theVertexPoint(aVertex), 
  theNorm(100.), 
  theX0(0.), 
  theY0(0.), 
  theRho(0.),
  theN1(0.),
  theN2(0.),
  theC(0.),
  theValid(true) {

  createCircleParameters();
  
}

FastCircle::FastCircle(const GlobalPoint& outerHit,
		       const GlobalPoint& middleHit,
		       const GlobalPoint& aVertex,
		       double norm) : 
  theOuterPoint(outerHit), 
  theInnerPoint(middleHit), 
  theVertexPoint(aVertex), 
  theNorm(norm), 
  theX0(0.), 
  theY0(0.), 
  theRho(0.),
  theN1(0.),
  theN2(0.),
  theC(0.),
  theValid(true) {

  createCircleParameters();
  
}

void FastCircle::createCircleParameters() {
  
  AlgebraicVector x = transform(theOuterPoint);
  AlgebraicVector y = transform(theInnerPoint);
  AlgebraicVector z = transform(theVertexPoint);

  AlgebraicVector n(3,0);

  n(1) =   x(2)*(y(3) - z(3)) + y(2)*(z(3) - x(3)) + z(2)*(x(3) - y(3));
  n(2) = -(x(1)*(y(3) - z(3)) + y(1)*(z(3) - x(3)) + z(1)*(x(3) - y(3)));
  n(3) =   x(1)*(y(2) - z(2)) + y(1)*(z(2) - x(2)) + z(1)*(x(2) - y(2));

  n /= n.norm();
  double  c = -(n(1)*x(1) + n(2)*x(2) + n(3)*x(3));
  //  c = -(n(1)*y(1) + n(2)*y(2) + n(3)*y(3));
  //  c = -(n(1)*z(1) + n(2)*z(2) + n(3)*z(3));
  
  theN1 = n(1);
  theN2 = n(2);
  theC = c;

  if(fabs(c + n(3)) < 1.e-5) {
    // numeric limit
    // circle is more a straight line...
    theValid = false;
    return;
  }

  double x0 = -n(1) / (2.*(c + n(3)));
  double y0 = -n(2) / (2.*(c + n(3)));
  double rho = 
    sqrt((n(1)*n(1) + n(2)*n(2) - 4.*c*(c + n(3)))) / fabs(2.*(c + n(3)));
  
  theX0 = theNorm*x0;
  theY0 = theNorm*y0;
  theRho = theNorm*rho;

}

AlgebraicVector FastCircle::transform(const GlobalPoint& aPoint) const {

  AlgebraicVector riemannPoint(3,0);

  double R = aPoint.perp();
  R /= theNorm;
  double phi = 0.;
  if(R > 0.) phi = aPoint.phi();
  
  riemannPoint(1) = (R*cos(phi))/(1 + R*R);
  riemannPoint(2) = (R*sin(phi))/(1 + R*R);
  riemannPoint(3) = (R*R)/(1 + R*R);
  
  return riemannPoint;
}
