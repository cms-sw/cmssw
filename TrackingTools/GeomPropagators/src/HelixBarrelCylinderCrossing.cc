#include "TrackingTools/GeomPropagators/interface/HelixBarrelCylinderCrossing.h"
#include "TrackingTools/GeomPropagators/src/RealQuadEquation.h"
#include "TrackingTools/GeomPropagators/interface/StraightLineCylinderCrossing.h"

#include "DataFormats/GeometrySurface/interface/Cylinder.h"

#include <iostream>
#include <cmath>

#include<tuple>

template <typename T> 
inline T sqr(const T& t) {return t*t;}

HelixBarrelCylinderCrossing::
HelixBarrelCylinderCrossing( const GlobalPoint& startingPos,
			     const GlobalVector& startingDir,
			     double rho, PropagationDirection propDir, 
			     const Cylinder& cyl, Solution sol)
{
  // assumes the cylinder is centered at 0,0
  double R = cyl.radius();

  // protect for zero curvature case
  const double sraightLineCutoff = 1.e-7;
  if (fabs(rho)*R < sraightLineCutoff && 
      fabs(rho)*startingPos.perp()  < sraightLineCutoff) {
    // switch to straight line case
    StraightLineCylinderCrossing slc( cyl.toLocal(startingPos), 
				      cyl.toLocal(startingDir), propDir);
    std::pair<bool,double> pl = slc.pathLength( cyl);
    if (pl.first) {
      theSolExists = true;
      theS = pl.second;
      thePos = cyl.toGlobal(slc.position(theS));
      theDir = startingDir;
    }
    else theSolExists = false;
    return; // all needed data members have been set
  }

  double R2cyl = R*R;
  double pt   = startingDir.perp();
  Point center( startingPos.x()-startingDir.y()/(pt*rho),
		startingPos.y()+startingDir.x()/(pt*rho));
  double p2 = startingPos.perp2();
  bool solveForX;
  double B, C, E, F;
  if (fabs(center.x()) > fabs(center.y())) {
    solveForX = false;
    E = (R2cyl - p2) / (2.*center.x());
    F = center.y()/center.x();
    B = 2.*( startingPos.y() - F*startingPos.x() - E*F);
    C = 2.*E*startingPos.x() + E*E + p2 - R2cyl;
  }
  else {
    solveForX = true;
    E = (R2cyl - p2) / (2.*center.y());
    F = center.x()/center.y();
    B = 2.*( startingPos.x() - F*startingPos.y() - E*F);
    C = 2.*E*startingPos.y() + E*E + p2 - R2cyl;
  }

  RealQuadEquation eq( 1+F*F, B, C);
  if (!eq.hasSolution) {
    theSolExists = false;
    return;
  }

  Vector d1, d2;;
  if (solveForX) {
    d1 = Point(eq.first,  E-F*eq.first);
    d2 = Point(eq.second, E-F*eq.second);
  }
  else {
    d1 = Point( E-F*eq.first,  eq.first);
    d2 = Point( E-F*eq.second, eq.second);
  }
  
  Vector         theD;
  int            theActualDir;


  std::tie(theD,theActualDir) = chooseSolution(d1, d2, startingPos, startingDir, propDir);
  if (!theSolExists) return;

  float ipabs = 1.f/startingDir.mag();
  float sinTheta = float(pt) * ipabs;
  float cosTheta = startingDir.z() * ipabs;



  // -------

  auto dMag = theD.mag();
  float tmp = 0.5f * float(dMag * rho);
  if (std::abs(tmp)>1.f) tmp = std::copysign(1.f,tmp);
  theS = theActualDir * 2.f* std::asin( tmp ) / (float(rho)*sinTheta);
  thePos =  GlobalPoint( startingPos.x() + theD.x(),
			 startingPos.y() + theD.y(),
			 startingPos.z() + theS*cosTheta);

  if (sol==onlyPos) return;

  if (theS < 0) tmp = -tmp;
  auto sinPhi = 2.f*tmp*sqrt(1.f-tmp*tmp);
  auto cosPhi = 1.f-2.f*tmp*tmp;
  theDir = DirectionType(startingDir.x()*cosPhi-startingDir.y()*sinPhi,
			 startingDir.x()*sinPhi+startingDir.y()*cosPhi,
			 startingDir.z());

  if (sol!=bothSol) return;


  //-----  BM  
  //double momProj1 = startingDir.x()*d1.x() + startingDir.y()*d1.y();
  //double momProj2 = startingDir.x()*d2.x() + startingDir.y()*d2.y();

 
  int theActualDir1 = propDir==alongMomentum ? 1 : -1;
  int theActualDir2 = propDir==alongMomentum ? 1 : -1;


  auto dMag1 = d1.mag();
  auto tmp1 = 0.5f * dMag1 * float(rho);
  if (std::abs(tmp1)>1.f) tmp1 = std::copysign(1.f,tmp1);
  auto theS1 = theActualDir1 * 2.f* std::asin( tmp1 ) / (rho*sinTheta);
  thePos1 =  GlobalPoint( startingPos.x() + d1.x(),
			  startingPos.y() + d1.y(),
			  startingPos.z() + theS1*cosTheta);


  auto dMag2 = d2.mag();
  auto tmp2 = 0.5f * dMag2 * float(rho);
  if (std::abs(tmp2)>1.f) tmp2 = std::copysign(1.f,tmp2);
  auto theS2 = theActualDir2 * 2.f* std::asin( tmp2 ) / (float(rho)*sinTheta);
  thePos2 =  GlobalPoint( startingPos.x() + d2.x(),
			  startingPos.y() + d2.y(),
			  startingPos.z() + theS2*cosTheta);	

}

std::pair<HelixBarrelCylinderCrossing::Vector,int> HelixBarrelCylinderCrossing::chooseSolution( const Vector& d1, const Vector& d2,
						  const PositionType& startingPos,
						  const DirectionType& startingDir, 
						  PropagationDirection propDir)
{

  Vector         theD;
  int            theActualDir;

  auto momProj1 = startingDir.x()*d1.x() + startingDir.y()*d1.y();
  auto momProj2 = startingDir.x()*d2.x() + startingDir.y()*d2.y();

  if ( propDir == anyDirection ) {
    theSolExists = true;
    if (d1.mag2()<d2.mag2()) {
      theD = d1;
      theActualDir = (momProj1 > 0) ? 1 : -1;
    }
    else {
      theD = d2;
      theActualDir = (momProj2 > 0) ? 1 : -1;
    }
  }
  else {
    int propSign = propDir==alongMomentum ? 1 : -1;
    if (momProj1*momProj2 < 0) {
      // if different signs return the positive one
      theSolExists = true;
      theD = (momProj1*propSign > 0) ? d1 : d2;
      theActualDir = propSign;
    }
    else if (momProj1*propSign > 0) {
      // if both positive, return the shortest
      theSolExists = true;
      theD = (d1.mag2()<d2.mag2()) ? d1 : d2;
      theActualDir = propSign;
    }
    else theSolExists = false;
  }

  return std::pair<Vector,int>(theD,theActualDir);
}

