#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkSeedGenerator/interface/FastLine.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

FastHelix::FastHelix(const GlobalPoint& outerHit,
		     const GlobalPoint& middleHit,
		     const GlobalPoint& aVertex,
		     const edm::EventSetup& iSetup) : theOuterHit(outerHit),
						      theMiddleHit(middleHit),
						      theVertex(aVertex),
						      theCircle(outerHit,
								middleHit,
								aVertex) {

		       iSetup.get<IdealMagneticFieldRecord>().get(pSetup);
		       tesla0=pSetup->inTesla(GlobalPoint(0,0,0));
		     }

FreeTrajectoryState FastHelix::stateAtVertex() const {
  
  if(isValid() && (fabs(tesla0.z()) > 1e-3))
    return helixStateAtVertex();
  else 
    return straightLineStateAtVertex();
    
}

FreeTrajectoryState FastHelix::helixStateAtVertex() const {
  
  GlobalPoint pMid(theMiddleHit);
  GlobalPoint v(theVertex);
  
  double dydx = 0.;
  double pt = 0., px = 0., py = 0.;
  
  //remember (radius rho in cm):
  //rho = 
  //100. * pt * 
  //(10./(3.*MagneticField::inTesla(GlobalPoint(0., 0., 0.)).z()));
  
  double rho = theCircle.rho();
  // pt = 0.01 * rho * (0.3*MagneticField::inTesla(GlobalPoint(0.,0.,0.)).z());
  pt = 0.01 * rho * (0.3*tesla0.z());
  //  pt = 0.01 * rho * (0.3*GlobalPoint(0.,0.,0.).MagneticField().z());

  // (py/px)|x=v.x() = (dy/dx)|x=v.x()
  //remember:
  //y(x) = +-sqrt(rho^2 - (x-x0)^2) + y0 
  //y(x) =  sqrt(rho^2 - (x-x0)^2) + y0  if y(x) >= y0 
  //y(x) = -sqrt(rho^2 - (x-x0)^2) + y0  if y(x) < y0
  //=> (dy/dx) = -(x-x0)/sqrt(Q)  if y(x) >= y0
  //   (dy/dx) =  (x-x0)/sqrt(Q)  if y(x) < y0
  //with Q = rho^2 - (x-x0)^2
  double root = sqrt(rho*rho - (v.x()-theCircle.x0())*(v.x()-theCircle.x0()));
  if((v.y() - theCircle.y0()) > 0.)
    dydx = -(v.x() - theCircle.x0()) / root;
  else
    dydx = (v.x() - theCircle.x0()) / root;
  
  px = pt/sqrt(1. + dydx*dydx);
  py = px*dydx;
  // check sign with scalar product
  if(px*(pMid.x() - v.x()) + py*(pMid.y() - v.y()) < 0.) {
    px *= -1.;
    py *= -1.;
  } 

  //calculate z0, pz
  //(z, R*phi) linear relation in a helix
  //with R, phi defined as radius and angle w.r.t. centre of circle
  //in transverse plane
  //pz = pT*(dz/d(R*phi)))
  
  FastLine flfit(theOuterHit, theMiddleHit, theCircle.rho());
  double z_0 = -flfit.c()/flfit.n2();
  double dzdrphi = -flfit.n1()/flfit.n2();
  double pz = pt*dzdrphi;

  //get sign of particle



  GlobalVector magvtx=pSetup->inTesla(v);
  TrackCharge q = 
    ((theCircle.x0()*py - theCircle.y0()*px) / 
     (magvtx.z()) < 0.) ? 
    -1 : 1;
  
  AlgebraicSymMatrix C(5,1);
  //MP
  FTS atVertex = FTS(GlobalTrajectoryParameters(GlobalPoint(v.x(), v.y(), z_0),
						GlobalVector(px, py, pz),
						q,
						&(*pSetup)),
		     CurvilinearTrajectoryError(C));
  

  
  return atVertex;
}

FreeTrajectoryState FastHelix::straightLineStateAtVertex() const {

  //calculate FTS assuming straight line...

  GlobalPoint pMid(theMiddleHit);
  GlobalPoint v(theVertex);

  double dydx = 0.;
  double pt = 0., px = 0., py = 0.;
  
  if(fabs(theCircle.n1()) > 0. || fabs(theCircle.n2()) > 0.)
    pt = 1.e+4;// 10 TeV //else no pt
  if(fabs(theCircle.n2()) > 0.) {
    dydx = -theCircle.n1()/theCircle.n2(); //else px = 0 
  }
  px = pt/sqrt(1. + dydx*dydx);
  py = px*dydx;
  // check sign with scalar product
  if (px*(pMid.x() - v.x()) + py*(pMid.y() - v.y()) < 0.) {
    px *= -1.;
    py *= -1.;
  } 

  //calculate z_0 and pz at vertex using weighted mean
  //z = z(r) = z0 + (dz/dr)*r
  //tan(theta) = dr/dz = (dz/dr)^-1
  //theta = atan(1./dzdr)
  //p = pt/sin(theta)
  //pz = p*cos(theta) = pt/tan(theta) 

  FastLine flfit(theOuterHit, theMiddleHit);
  double z_0 = -flfit.c()/flfit.n2();
  double dzdr = -flfit.n1()/flfit.n2();
  double pz = pt*dzdr; 
  
  TrackCharge q = 1;
  AlgebraicSymMatrix66 C = AlgebraicMatrixID();
  //MP
  FTS atVertex = FTS(GlobalTrajectoryParameters(GlobalPoint(v.x(), v.y(), z_0),
						GlobalVector(px, py, pz),
						q,
						&(*pSetup)),
		     CartesianTrajectoryError());
  
  return atVertex;
}
