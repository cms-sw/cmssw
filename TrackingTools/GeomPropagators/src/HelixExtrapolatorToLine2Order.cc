#include "TrackingTools/GeomPropagators/interface/HelixExtrapolatorToLine2Order.h"
#include "DataFormats/GeometrySurface/interface/Line.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <cfloat>

HelixExtrapolatorToLine2Order::HelixExtrapolatorToLine2Order(const PositionType& point,
							     const DirectionType& direction,
							     const float curvature,
							     const PropagationDirection propDir) :
  thePosition(point),
  theRho(curvature),
  thePropDir(propDir)
{
  //
  // Components of direction vector (with correct normalisation)
  //
  double px = direction.x();
  double py = direction.y();
  double pz = direction.z();
  double pt = px*px+py*py;
  double p = sqrt(pt+pz*pz);
  pt = sqrt(pt);
  theDirection = DirectionTypeDouble(px/pt,py/pt,pz/pt);
  theSinTheta = pt/p;
}

//
// Propagation status and path length to closest approach with point
//
std::pair<bool,double>
HelixExtrapolatorToLine2Order::pathLength (const GlobalPoint& point) const {
  //
  PositionTypeDouble position(point);
  DirectionTypeDouble helix2(-0.5*theRho*theDirection.y(),
			     0.5*theRho*theDirection.x(),
			     0.);
  DirectionTypeDouble deltaPos(thePosition-position);
  //
  // coefficients of 3rd order equation
  //
  double ceq[4];
  ceq[3] = 2*helix2.mag2();
  // ceq[2] = 3*theDirection.dot(helix2) = 0 since they are orthogonal 
  ceq[2] = 0.;
  ceq[1] = theDirection.mag2()+2*deltaPos.dot(helix2);
  ceq[0] = deltaPos.dot(theDirection);
  //
  return pathLengthFromCoefficients(ceq);
}

//
// Propagation status and path length to closest approach with line
//
std::pair<bool,double>
HelixExtrapolatorToLine2Order::pathLength (const Line& line) const {
  //
  // Auxiliary vectors. Assumes that line.direction().mag()=1 !
  //
  PositionTypeDouble linePosition(line.position());
  DirectionTypeDouble lineDirection(line.direction());
  DirectionTypeDouble helix2(-0.5*theRho*theDirection.y(),
			     0.5*theRho*theDirection.x(),
			     0.);
  DirectionTypeDouble deltaPos(thePosition-linePosition);
  DirectionTypeDouble helix1p(theDirection-lineDirection*theDirection.dot(lineDirection));
  DirectionTypeDouble helix2p(helix2-lineDirection*helix2.dot(lineDirection));
  //
  // coefficients of 3rd order equation
  //
  double ceq[4];
  ceq[3] = 2*helix2.dot(helix2p);
  // ceq[2] = 3*helix1.dot(helix1p); 
  // since theDirection.dot(helix2)==0 equivalent to
  ceq[2] = 3*theDirection.dot(lineDirection)*helix2.dot(lineDirection);
  ceq[1] = theDirection.dot(helix1p)+2*deltaPos.dot(helix2p);
  ceq[0] = deltaPos.dot(helix1p);
  //
  return pathLengthFromCoefficients(ceq);
}

//
// Propagation status and path length to intersection
//
std::pair<bool,double>
HelixExtrapolatorToLine2Order::pathLengthFromCoefficients (const double ceq[4]) const 
{
  //
  // Solution of 3rd order equation
  //
  double solutions[3];
  unsigned int nRaw = solve3rdOrder(ceq,solutions);
  //
  // check compatibility with propagation direction
  //
  unsigned int nDir(0);
  for ( unsigned int i=0; i<nRaw; i++ ) {
    if ( thePropDir==anyDirection ||
	 (solutions[i]>=0&&thePropDir==alongMomentum) ||
	 (solutions[i]<=0&&thePropDir==oppositeToMomentum) )
      solutions[nDir++] = solutions[i];
  }
  if ( nDir==0 )  return std::make_pair(false,0.);
  //
  // check 2nd derivative
  //
  unsigned int nMin(0);
  for ( unsigned int i=0; i<nDir; i++ ) {
    double st = solutions[i];
    double deri2 = (3*ceq[3]*st+2*ceq[2])*st+ceq[1];
    if ( deri2>0. )  solutions[nMin++] = st;
  }
  if ( nMin==0 )  return std::make_pair(false,0.);
  //
  // choose smallest path length
  //
  double dSt = solutions[0];
  for ( unsigned int i=1; i<nMin; i++ ) {
    if ( fabs(solutions[i])<fabs(dSt) )  dSt = solutions[i];
  }

  return std::make_pair(true,dSt/theSinTheta);
}

int
HelixExtrapolatorToLine2Order::solve3rdOrder (const double ceq[], 
					      double solutions[]) const
{
  //
  // Real 3rd order equation? Follow numerical recipes ..
  //
  if ( fabs(ceq[3])>FLT_MIN ) {
    int result(0);
    double q = (ceq[2]*ceq[2]-3*ceq[3]*ceq[1]) / (ceq[3]*ceq[3]) / 9.;
    double r = (2*ceq[2]*ceq[2]*ceq[2]-9*ceq[3]*ceq[2]*ceq[1]+27*ceq[3]*ceq[3]*ceq[0]) 
      / (ceq[3]*ceq[3]*ceq[3]) / 54.;
    double q3 = q*q*q;
    if ( r*r<q3 ) {
      double phi = acos(r/sqrt(q3))/3.;
      double rootq = sqrt(q);
      for ( int i=0; i<3; i++ ) {
	solutions[i] = -2*rootq*cos(phi) - ceq[2]/ceq[3]/3.;
	phi += 2./3.*M_PI;
      }
      result = 3;
    }
    else {
      double a = pow(fabs(r)+sqrt(r*r-q3),1./3.);
      if ( r>0. ) a *= -1;
      double b = fabs(a)>FLT_MIN ? q/a : 0.;
      solutions[0] = a + b - ceq[2]/ceq[3]/3.;
      result = 1;
    }
    return result;
  }
  //
  // Second order equation
  //
  else if ( fabs(ceq[2])>FLT_MIN ) {
    return solve2ndOrder(ceq,solutions);
  }
  else {
    //
    // Special case: linear equation
    //
    solutions[0] = -ceq[0]/ceq[1];
    return 1;
  }
}

int
HelixExtrapolatorToLine2Order::solve2ndOrder (const double coeff[], 
					      double solutions[]) const
{
  //
  double deq1 = coeff[1]*coeff[1];
  double deq2 = coeff[2]*coeff[0];
  if ( fabs(deq1)<FLT_MIN || fabs(deq2/deq1)>1.e-6 ) {
    //
    // Standard solution for quadratic equations
    //
    double deq = deq1+2*deq2;
    if ( deq<0. )  return 0;
    double ceq = -0.5*(coeff[1]+(coeff[1]>0?1:-1)*sqrt(deq));
    solutions[0] = -2*ceq/coeff[2];
    solutions[1] = coeff[0]/ceq;
    return 2;
  }
  else {
    //
    // Solution by expansion of sqrt(1+deq)
    //
    double ceq = coeff[1]/coeff[2];
    double deq = deq2/deq1;
    deq *= (1-deq/2);
    solutions[0] = -ceq*deq;
    solutions[1] = ceq*(2+deq);
    return 2;
  }
}
//
// Position after a step of path length s (2nd order)
//
HelixLineExtrapolation::PositionType
HelixExtrapolatorToLine2Order::position (double s) const {
  // use double precision result
//   PositionTypeDouble pos = positionInDouble(s);
//   return PositionType(pos.x(),pos.y(),pos.z());
  return PositionType(positionInDouble(s));
}
//
// Position after a step of path length s (2nd order) (in double precision)
//
HelixExtrapolatorToLine2Order::PositionTypeDouble
HelixExtrapolatorToLine2Order::positionInDouble (double s) const {
  // based on path length in the transverse plane
  double st = s*theSinTheta;
  return PositionTypeDouble(thePosition.x()+(theDirection.x()-st*0.5*theRho*theDirection.y())*st,
			    thePosition.y()+(theDirection.y()+st*0.5*theRho*theDirection.x())*st,
			    thePosition.z()+st*theDirection.z());
}
//
// Direction after a step of path length 2 (2nd order) (in double precision)
//
HelixLineExtrapolation::DirectionType
HelixExtrapolatorToLine2Order::direction (double s) const {
  // use double precision result
//   DirectionTypeDouble dir = directionInDouble(s);
//   return DirectionType(dir.x(),dir.y(),dir.z());
   return DirectionType(directionInDouble(s));
}
//
// Direction after a step of path length 2 (2nd order)
//
HelixExtrapolatorToLine2Order::DirectionTypeDouble
HelixExtrapolatorToLine2Order::directionInDouble (double s) const {
  // based on delta phi
  double dph = s*theRho*theSinTheta;
  return DirectionTypeDouble(theDirection.x()-(theDirection.y()+0.5*theDirection.x()*dph)*dph,
			     theDirection.y()+(theDirection.x()-0.5*theDirection.y()*dph)*dph,
			     theDirection.z());
}
