#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "RecoTracker/TkSeedGenerator/interface/FastLine.h"

void FastHelix::compute() {
  
  if(isValid() && (std::abs(tesla0) > 1e-3) && theCircle.rho()<maxRho)
    helixStateAtVertex();
  else 
    straightLineStateAtVertex();
    
}

void FastHelix::helixStateAtVertex() {

  // given the above rho>0.
  double rho = theCircle.rho();
  //remember (radius rho in cm):
  //rho = 
  //100. * pt * 
  //(10./(3.*MagneticField::inTesla(GlobalPoint(0., 0., 0.)).z()));
  
  // pt = 0.01 * rho * (0.3*MagneticField::inTesla(GlobalPoint(0.,0.,0.)).z());
  double cm2GeV = 0.01 * 0.3*tesla0;
  double pt = cm2GeV * rho; 
 
  // verify that rho is not toooo large
  double dcphi = ((outerHit().x()-theCircle.x0())*(middleHit().x()-theCircle.x0()) +
		 (outerHit().y()-theCircle.y0())*(middleHit().y()-theCircle.y0())
		 )/(rho*rho);
  if (std::abs(dcphi)>=1.f) { straightLineStateAtVertex(); return;}
  
  GlobalPoint pMid(middleHit());
  GlobalPoint v(vertex());
  
  // tangent in v (or the opposite...)
  double px = -cm2GeV * (v.y()-theCircle.y0());
  double py =  cm2GeV * (v.x()-theCircle.x0());
  // check sign with scalar product
  if(px*(pMid.x() - v.x()) + py*(pMid.y() - v.y()) < 0.) {
    px = -px;
    py = -py;
  } 

 
 
  //calculate z0, pz
  //(z, R*phi) linear relation in a helix
  //with R, phi defined as radius and angle w.r.t. centre of circle
  //in transverse plane
  //pz = pT*(dz/d(R*phi)))
  

  // VI 23/01/2012
  double dzdrphi = outerHit().z() - middleHit().z();
  dzdrphi /= rho*acos(dcphi);
  double pz = pt*dzdrphi;


  TrackCharge q = 1;
  if (theCircle.x0()*py - theCircle.y0()*px < 0) q =-q;
  if (tesla0 < 0.) q =-q;

  //VI
  if ( useBasisVertex ) {
    atVertex =  GlobalTrajectoryParameters(basisVertex, 
					   GlobalVector(px, py, pz),
					   q, 
					   bField
					   );
  } else {
    double z_0 =  middleHit().z();
    // assume v is before middleHit (opposite to outer)
    double ds = ( (v.x()-theCircle.x0())*(middleHit().x()-theCircle.x0()) +
		  (v.y()-theCircle.y0())*(middleHit().y()-theCircle.y0())
		  )/(rho*rho);
    if (std::abs(ds)<1.f) {
      ds = rho*acos(ds);
      z_0 -= ds*dzdrphi;
    } else { // line????
      z_0 -= std::sqrt((middleHit()-v).perp2()/(outerHit()-middleHit()).perp2())*(outerHit().z()-middleHit().z());
    }
    
    //double z_old = -flfit.c()/flfit.n2();
    // std::cout << "v:xyz, z,old,new " << v << "   " << z_old << " " << z_0 << std::endl;

    atVertex =  GlobalTrajectoryParameters(GlobalPoint(v.x(),v.y(),z_0), 
					   GlobalVector(px, py, pz),
					   q, 
					   bField
					   );
  }
  
}

void FastHelix::straightLineStateAtVertex() {

  //calculate GlobalTrajectoryParameters assuming straight line...

  GlobalPoint pMid(middleHit());
  GlobalPoint v(vertex());

  double dydx = 0.;
  double pt = 0., px = 0., py = 0.;
  
  if(fabs(theCircle.n1()) > 0. || fabs(theCircle.n2()) > 0.)
    pt = maxPt  ;// 10 TeV //else no pt
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

  FastLine flfit(outerHit(), middleHit());
  double dzdr = -flfit.n1()/flfit.n2();
  double pz = pt*dzdr; 
  
  TrackCharge q = 1;
  //VI

  if ( useBasisVertex ) {
    atVertex = GlobalTrajectoryParameters(basisVertex, 
					  GlobalVector(px, py, pz),
					  q, 
					  bField
					  );
  } else {
  double z_0 = -flfit.c()/flfit.n2();
  atVertex = GlobalTrajectoryParameters(GlobalPoint(v.x(), v.y(), z_0),
					GlobalVector(px, py, pz),
					q,
					bField
					);
  }
}
