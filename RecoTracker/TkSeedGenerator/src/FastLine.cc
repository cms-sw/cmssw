#include "RecoTracker/TkSeedGenerator/interface/FastLine.h"

FastLine::FastLine(const GlobalPoint& outerHit,
		   const GlobalPoint& innerHit) : 
  theOuterHit(outerHit),
  theInnerHit(innerHit),
  theRho(0.),
  theN1(0.),
  theN2(0.),
  theC(0.),
  theValid(true) {

  createLineParameters();
  
}

FastLine::FastLine(const GlobalPoint& outerHit,
		   const GlobalPoint& innerHit,
		   double rho) : 
  theOuterHit(outerHit),
  theInnerHit(innerHit),
  theRho(rho),
  theN1(0.),
  theN2(0.),
  theC(0.),
  theValid(true) {

  createLineParameters();
  
}

void FastLine::createLineParameters() {

  double rphi0 = 0., rphi1 = 0.;
  if(theRho > 0.) {
    rphi0 = theRho*acos(1. - theInnerHit.perp2()/(2.*theRho*theRho));
    rphi1 = theRho*acos(1. - theOuterHit.perp2()/(2.*theRho*theRho));
  } else {
    rphi0 = theInnerHit.perp();
    rphi1 = theOuterHit.perp();
  }

  double n1 = theInnerHit.z() - theOuterHit.z();
  double n2 = -(rphi0 - rphi1);
  double norm = sqrt(n1*n1 + n2*n2);
  theN1 = n1/norm;
  theN2= n2/norm;
  theC = -(theN1*rphi0 + theN2*theInnerHit.z());
  //  theC = -(theN1*rphi1 + theN2*theOuterHit.z());

}






