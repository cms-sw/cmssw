#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

FastCircle::FastCircle(const GlobalPoint& outerHit,
		       const GlobalPoint& middleHit,
		       const GlobalPoint& aVertex) : 
  theOuterPoint(outerHit), 
  theInnerPoint(middleHit), 
  theVertexPoint(aVertex), 
  theNorm(128.), 
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

namespace {
  inline
  AlgebraicVector3 transform(const GlobalPoint& aPoint, float norm) {
    
    AlgebraicVector3 riemannPoint;
    
    auto p = aPoint.basicVector()/norm;
    float R2 = p.perp2();
    float fact = 1.f/(1.f+R2); // let's factorize the common factor out
    riemannPoint[0] = fact*p.x();  
    riemannPoint[1] = fact*p.y();  
    riemannPoint[2] = fact*R2;
    
    return riemannPoint;
  }


}

void FastCircle::createCircleParameters() {
  
  AlgebraicVector3 x = transform(theOuterPoint,theNorm);
  AlgebraicVector3 y = transform(theInnerPoint,theNorm);
  AlgebraicVector3 z = transform(theVertexPoint,theNorm);

  AlgebraicVector3 n;

  n[0] =   x[1]*(y[2] - z[2]) + y[1]*(z[2] - x[2]) + z[1]*(x[2] - y[2]);
  n[1] = -(x[0]*(y[2] - z[2]) + y[0]*(z[2] - x[2]) + z[0]*(x[2] - y[2]));
  n[2] =   x[0]*(y[1] - z[1]) + y[0]*(z[1] - x[1]) + z[0]*(x[1] - y[1]);

  double mag2 = n[0]*n[0]+n[1]*n[1]+n[2]*n[2];
  if (mag2 < 1.e-20) {
    theValid = false;
    return;
  }
  n.Unit(); // reduce n to a unit vector
  double  c = -(n[0]*x[0] + n[1]*x[1] + n[2]*x[2]);
  //  c = -(n[0]*y[0] + n[1]*y[1] + n[2]*y[2]);
  //  c = -(n[0]*z[0] + n[1]*z[1] + n[2]*z[2]);
  
  theN1 = n[0];
  theN2 = n[1];
  theC = c;

  if(fabs(c + n[2]) < 1.e-5) {
    // numeric limit
    // circle is more a straight line...
    theValid = false;
    return;
  }

  double x0 = -n[0] / (2.*(c + n[2]));
  double y0 = -n[1] / (2.*(c + n[2]));
  double rho = 
    sqrt((n[0]*n[0] + n[1]*n[1] - 4.*c*(c + n[2]))) / fabs(2.*(c + n[2]));
  
  theX0 = theNorm*x0;
  theY0 = theNorm*y0;
  theRho = theNorm*rho;

}

