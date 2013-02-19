#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"
#include "TrackingTools/GeomPropagators/src/RealQuadEquation.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/SIMDVec.h"

#include <algorithm>
#include <cfloat>

HelixBarrelPlaneCrossingByCircle::
HelixBarrelPlaneCrossingByCircle( const PositionType& pos,
				  const DirectionType& dir,
				  double rho, PropagationDirection propDir) :
  theStartingPos( pos), theStartingDir(dir), 
  theRho(rho), thePropDir(propDir) { init();}

HelixBarrelPlaneCrossingByCircle::
HelixBarrelPlaneCrossingByCircle( const GlobalPoint& pos,
				  const GlobalVector& dir,
				  double rho, PropagationDirection propDir) :
  theStartingPos( pos.basicVector()), theStartingDir(dir.basicVector()), 
  theRho(rho), thePropDir(propDir) { init();}

void HelixBarrelPlaneCrossingByCircle::init()
{
  double pabsI = 1./theStartingDir.mag();
  double pt   = theStartingDir.perp();
  theCosTheta = theStartingDir.z()*pabsI;
  theSinTheta = pt*pabsI;

  // protect for zero curvature case
  const double sraightLineCutoff = 1.e-7;
  if (fabs(theRho) < sraightLineCutoff && 
      fabs(theRho)*theStartingPos.perp()  < sraightLineCutoff) {
    useStraightLine = true;
  }else{
    // circle parameters
    // position of center of curvature is on a line perpendicular
    // to startingDir and at a distance 1/curvature.
    double o = 1./(pt*theRho);
    theXCenter = theStartingPos.x() - theStartingDir.y()*o;
    theYCenter = theStartingPos.y() + theStartingDir.x()*o;
    useStraightLine = false;
  }
}

std::pair<bool,double>
HelixBarrelPlaneCrossingByCircle::pathLength( const Plane& plane)
{
  typedef std::pair<bool,double>     ResultType;
  
  if(useStraightLine){
    // switch to straight line case
    StraightLinePlaneCrossing slc( theStartingPos, 
				   theStartingDir, thePropDir);
    return slc.pathLength( plane);
  }
  

  // plane parameters
  GlobalVector n = plane.normalVector();
  double distToPlane = -plane.localZ( GlobalPoint( theStartingPos));
  //    double distToPlane = (plane.position().x()-theStartingPos.x()) * n.x() +
  //                         (plane.position().y()-theStartingPos.y()) * n.y();
  double nx = n.x();  // convert to double
  double ny = n.y();  // convert to double
  double distCx = theStartingPos.x() - theXCenter;
  double distCy = theStartingPos.y() - theYCenter;

  double nfac, dfac;
  double A, B, C;
  bool solveForX;
  if (fabs(nx) > fabs(ny)) {
    solveForX = false;
    nfac = ny/nx;
    dfac = distToPlane/nx;
    B = distCy - nfac*distCx;  // only part of B, may have large cancelation
    C = (2.*distCx + dfac)*dfac;
  }
  else {
    solveForX = true;
    nfac = nx/ny;
    dfac = distToPlane/ny;
    B = distCx - nfac*distCy; // only part of B, may have large cancelation
    C = (2.*distCy + dfac)*dfac;
  }
  B -= nfac*dfac; B *= 2;  // the rest of B (normally small)
  A = 1.+ nfac*nfac;

  double dx1, dx2, dy1, dy2;
  RealQuadEquation equation(A,B,C);
  if (!equation.hasSolution) return ResultType( false, 0.);
  else {
    if (solveForX) {
      dx1 = equation.first;
      dx2 = equation.second;
      dy1 = dfac - nfac*dx1;
      dy2 = dfac - nfac*dx2;
    }
    else {
      dy1 = equation.first;
      dy2 = equation.second;
      dx1 = dfac - nfac*dy1;
      dx2 = dfac - nfac*dy2;
    }
  }
  bool solved = chooseSolution( Vector2D(dx1, dy1), Vector2D(dx2, dy2));
  if (solved) {
    theDmag = theD.mag();
    // protect asin (taking some safety margin)
    double sinAlpha = 0.5*theDmag*theRho;
    if ( sinAlpha>(1.-10*DBL_EPSILON) )  sinAlpha = 1.-10*DBL_EPSILON;
    else if ( sinAlpha<-(1.-10*DBL_EPSILON) )  sinAlpha = -(1.-10*DBL_EPSILON);
    theS = theActualDir*2./(theRho*theSinTheta) * asin( sinAlpha);
    return ResultType( true, theS);
  }
  else return ResultType( false, 0.);
}

bool
HelixBarrelPlaneCrossingByCircle::chooseSolution( const Vector2D& d1, 
						  const Vector2D& d2)
{
  bool solved;
  double momProj1 = theStartingDir.x()*d1.x() + theStartingDir.y()*d1.y();
  double momProj2 = theStartingDir.x()*d2.x() + theStartingDir.y()*d2.y();

  if ( thePropDir == anyDirection ) {
    solved = true;
    if (d1.mag2()<d2.mag2()) {
      theD = d1;
      theActualDir = (momProj1 > 0) ? 1. : -1.;
    }
    else {
      theD = d2;
      theActualDir = (momProj2 > 0) ? 1. : -1.;
    }

  }
  else {
    double propSign = thePropDir==alongMomentum ? 1 : -1;
    if (!mathSSE::samesign(momProj1,momProj2)) {
      // if different signs return the positive one
      solved = true;
      theD         = mathSSE::samesign(momProj1,propSign) ? d1 : d2;
      theActualDir = propSign;
    }
    else if (mathSSE::samesign(momProj1,propSign)) {
      // if both positive, return the shortest
      solved = true;
      theD = (d1.mag2()<d2.mag2()) ? d1 : d2;
      theActualDir = propSign;
    }
    else solved = false;
  }
  return solved;
}

HelixPlaneCrossing::PositionType 
HelixBarrelPlaneCrossingByCircle::position( double s) const
{
  if(useStraightLine){
    return PositionType(theStartingPos+s*theStartingDir.unit());
  }else{
    if ( s==theS) {
      return PositionType( theStartingPos.x() + theD.x(),
			   theStartingPos.y() + theD.y(), 
			   theStartingPos.z() + s*theCosTheta);
    }
    else {
      double phi = s*theSinTheta*theRho;
      double x1Shift = theStartingPos.x() - theXCenter;
      double y1Shift = theStartingPos.y() - theYCenter;
      
      return PositionType(x1Shift*cos(phi)-y1Shift*sin(phi) + theXCenter,
			  x1Shift*sin(phi)+y1Shift*cos(phi) + theYCenter,
			  theStartingPos.z() + s*theCosTheta);
    }
  }
}

HelixPlaneCrossing::DirectionType 
HelixBarrelPlaneCrossingByCircle::direction( double s) const
{
  if(useStraightLine){return theStartingDir;}
  else{
    double sinPhi, cosPhi;
    if ( s==theS) {
      double tmp = 0.5*theDmag*theRho;
      if (s < 0) tmp = -tmp;
      // protect sqrt
      sinPhi = 1.-(tmp*tmp);
      if ( sinPhi<0 )  sinPhi = 0.;
      sinPhi = 2.*tmp*sqrt(sinPhi);
      cosPhi = 1.-2.*(tmp*tmp);
    }
    else {
      double phi = s*theSinTheta*theRho;
      sinPhi = sin(phi);
      cosPhi = cos(phi);
    }
    return DirectionType(theStartingDir.x()*cosPhi-theStartingDir.y()*sinPhi,
			 theStartingDir.x()*sinPhi+theStartingDir.y()*cosPhi,
			 theStartingDir.z());
  }
}
