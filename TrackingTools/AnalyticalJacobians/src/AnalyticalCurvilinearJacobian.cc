#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

AnalyticalCurvilinearJacobian::AnalyticalCurvilinearJacobian 
(const GlobalTrajectoryParameters& globalParameters,
 const GlobalPoint& x, 
 const GlobalVector& p, 
 const double& s) :  theJacobian(AlgebraicMatrixID())
{
  //
  // helix: calculate full jacobian
  //
  if ( s*s*fabs(globalParameters.transverseCurvature())>1.e-5 ) { 
    GlobalPoint xStart = globalParameters.position();
    GlobalVector h  = globalParameters.magneticFieldInInverseGeV(xStart);
    computeFullJacobian(globalParameters,x,p,h,s);
  }
  //
  // straight line approximation, error in RPhi about 0.1um
  //
  else
    computeStraightLineJacobian(globalParameters,x,p,s);
   //dbg::dbg_trace(1,"ACJ1", globalParameters.vector(),x,p,s,theJacobian);
}


AnalyticalCurvilinearJacobian::AnalyticalCurvilinearJacobian
(const GlobalTrajectoryParameters& globalParameters,
 const GlobalPoint& x, 
 const GlobalVector& p, 
 const GlobalVector& h, // h is the magnetic Field in Inverse GeV
 const double& s) :  theJacobian(AlgebraicMatrixID())
{
  //
  // helix: calculate full jacobian
  //
  if ( s*s*fabs(globalParameters.transverseCurvature())>1.e-5 )
    computeFullJacobian(globalParameters,x,p,h,s);
  //
  // straight line approximation, error in RPhi about 0.1um
  //
  else
    computeStraightLineJacobian(globalParameters,x,p,s);
  
  //dbg::dbg_trace(1,"ACJ2", globalParameters.vector(),x,p,s,theJacobian);
}


#if defined(USE_SSEVECT) && !defined(TRPRFN_SCALAR)
#include "AnalyticalCurvilinearJacobianSSE.icc"
#elif defined(USE_EXTVECT) && !defined(TRPRFN_SCALAR)
#include "AnalyticalCurvilinearJacobianEXT.icc"

#else

void
AnalyticalCurvilinearJacobian::computeFullJacobian
(const GlobalTrajectoryParameters& globalParameters,
 const GlobalPoint& x, 
 const GlobalVector& p, 
 const GlobalVector& h, 
 const double& s)
{    
  //GlobalVector p1 = fts.momentum().unit();
  GlobalVector p1 = globalParameters.momentum().unit();
  GlobalVector p2 = p.unit();
  //GlobalPoint xStart = fts.position();
  GlobalPoint xStart = globalParameters.position();
  GlobalVector dx = xStart - x;
  //GlobalVector h  = MagneticField::inInverseGeV(xStart);
  // Martijn: field is now given as parameter.. GlobalVector h  = globalParameters.magneticFieldInInverseGeV(xStart);

  //double qbp = fts.signedInverseMomentum();
  double qbp = globalParameters.signedInverseMomentum();
  double absS = s;
  
  // calculate transport matrix
  // Origin: TRPRFN
  double t11 = p1.x(); double t12 = p1.y(); double t13 = p1.z();
  double t21 = p2.x(); double t22 = p2.y(); double t23 = p2.z();
  double cosl0 = p1.perp(); double cosl1 = 1./p2.perp();
  //AlgebraicMatrix a(5,5,1);
  // define average magnetic field and gradient 
  // at initial point - inlike TRPRFN
  GlobalVector hn = h.unit();
  double qp = -h.mag();
//   double q = -h.mag()*qbp;
  double q = qp*qbp;
  double theta = q*absS; double sint = sin(theta); double cost = cos(theta);
  double hn1 = hn.x(); double hn2 = hn.y(); double hn3 = hn.z();
  double dx1 = dx.x(); double dx2 = dx.y(); double dx3 = dx.z();
  double gamma = hn1*t21 + hn2*t22 + hn3*t23;
  double an1 = hn2*t23 - hn3*t22;
  double an2 = hn3*t21 - hn1*t23;
  double an3 = hn1*t22 - hn2*t21;
  double au = 1./sqrt(t11*t11 + t12*t12);
  double u11 = -au*t12; double u12 = au*t11;
  double v11 = -t13*u12; double v12 = t13*u11; double v13 = t11*u12 - t12*u11;
  au = 1./sqrt(t21*t21 + t22*t22);
  double u21 = -au*t22; double u22 = au*t21;
  double v21 = -t23*u22; double v22 = t23*u21; double v23 = t21*u22 - t22*u21;
  // now prepare the transport matrix
  // pp only needed in high-p case (WA)
//   double pp = 1./qbp;
////    double pp = fts.momentum().mag();
// moved up (where -h.mag() is needed()
//   double qp = q*pp;
  double anv = -(hn1*u21 + hn2*u22          );
  double anu =  (hn1*v21 + hn2*v22 + hn3*v23);
  double omcost = 1. - cost; double tmsint = theta - sint;
  
  double hu1 =         - hn3*u12;
  double hu2 = hn3*u11;
  double hu3 = hn1*u12 - hn2*u11;
  
  double hv1 = hn2*v13 - hn3*v12;
  double hv2 = hn3*v11 - hn1*v13;
  double hv3 = hn1*v12 - hn2*v11;
  
  //   1/p - doesn't change since |p1| = |p2|
  
  //   lambda
  
  theJacobian(1,0) = -qp*anv*(t21*dx1 + t22*dx2 + t23*dx3);
  
  theJacobian(1,1) = cost*(v11*v21 + v12*v22 + v13*v23) +
                      sint*(hv1*v21 + hv2*v22 + hv3*v23) +
                    omcost*(hn1*v11 + hn2*v12 + hn3*v13) *
                           (hn1*v21 + hn2*v22 + hn3*v23) +
                anv*(-sint*(v11*t21 + v12*t22 + v13*t23) +
                    omcost*(v11*an1 + v12*an2 + v13*an3) -
              tmsint*gamma*(hn1*v11 + hn2*v12 + hn3*v13) );

  theJacobian(1,2) = cost*(u11*v21 + u12*v22          ) +
                      sint*(hu1*v21 + hu2*v22 + hu3*v23) +
                    omcost*(hn1*u11 + hn2*u12          ) *
                           (hn1*v21 + hn2*v22 + hn3*v23) +
                anv*(-sint*(u11*t21 + u12*t22          ) +
                    omcost*(u11*an1 + u12*an2          ) -
              tmsint*gamma*(hn1*u11 + hn2*u12          ) );
  theJacobian(1,2) *= cosl0;

  theJacobian(1,3) = -q*anv*(u11*t21 + u12*t22          );

  theJacobian(1,4) = -q*anv*(v11*t21 + v12*t22 + v13*t23);

  //   phi

  theJacobian(2,0) = -qp*anu*(t21*dx1 + t22*dx2 + t23*dx3)*cosl1;

  theJacobian(2,1) = cost*(v11*u21 + v12*u22          ) +
                      sint*(hv1*u21 + hv2*u22          ) +
                    omcost*(hn1*v11 + hn2*v12 + hn3*v13) *
                           (hn1*u21 + hn2*u22          ) +
                anu*(-sint*(v11*t21 + v12*t22 + v13*t23) +
                    omcost*(v11*an1 + v12*an2 + v13*an3) -
              tmsint*gamma*(hn1*v11 + hn2*v12 + hn3*v13) );
  theJacobian(2,1) *= cosl1;

  theJacobian(2,2) = cost*(u11*u21 + u12*u22          ) +
                      sint*(hu1*u21 + hu2*u22          ) +
                    omcost*(hn1*u11 + hn2*u12          ) *
                           (hn1*u21 + hn2*u22          ) +
                anu*(-sint*(u11*t21 + u12*t22          ) +
                    omcost*(u11*an1 + u12*an2          ) -
              tmsint*gamma*(hn1*u11 + hn2*u12          ) );
  theJacobian(2,2) *= cosl1*cosl0;

  theJacobian(2,3) = -q*anu*(u11*t21 + u12*t22          )*cosl1;

  theJacobian(2,4) = -q*anu*(v11*t21 + v12*t22 + v13*t23)*cosl1;

  //   yt

  //double cutCriterion = abs(s/fts.momentum().mag());
  double cutCriterion = fabs(s/globalParameters.momentum().mag());
  const double limit = 5.; // valid for propagations with effectively float precision

  if (cutCriterion > limit) {
    double pp = 1./qbp;
    theJacobian(3,0) = pp*(u21*dx1 + u22*dx2            );
    theJacobian(4,0) = pp*(v21*dx1 + v22*dx2 + v23*dx3);
  }
  else {
    double hp11 = hn2*t13 - hn3*t12;
    double hp12 = hn3*t11 - hn1*t13;
    double hp13 = hn1*t12 - hn2*t11;
    double temp1 = hp11*u21 + hp12*u22;
    double s2 = s*s;
    double secondOrder41 = 0.5 * qp * temp1 * s2;
    double ghnmp1 = gamma*hn1 - t11;
    double ghnmp2 = gamma*hn2 - t12;
    double ghnmp3 = gamma*hn3 - t13;
    double temp2 = ghnmp1*u21 + ghnmp2*u22;
    double s3 = s2 * s;
    double s4 = s3 * s;
    double h1 = h.mag();
    double h2 = h1 * h1;
    double h3 = h2 * h1;
    double qbp2 = qbp * qbp;
    //                           s*qp*s* (qp*s *qbp)
    double thirdOrder41 = 1./3 * h2 * s3 * qbp * temp2;
    //                           -qp * s * qbp  * above
    double fourthOrder41 = 1./8 * h3 * s4 * qbp2 * temp1;
    theJacobian(3,0) = secondOrder41 + (thirdOrder41 + fourthOrder41);

    double temp3 = hp11*v21 + hp12*v22 + hp13*v23;
    double secondOrder51 = 0.5 * qp * temp3 * s2;
    double temp4 = ghnmp1*v21 + ghnmp2*v22 + ghnmp3*v23;
    double thirdOrder51 = 1./3 * h2 * s3 * qbp * temp4;
    double fourthOrder51 = 1./8 * h3 * s4 * qbp2 * temp3;
    theJacobian(4,0) = secondOrder51 + (thirdOrder51 + fourthOrder51);
  }

  theJacobian(3,1) = (sint*(v11*u21 + v12*u22          ) +
                     omcost*(hv1*u21 + hv2*u22          ) +
                     tmsint*(hn1*u21 + hn2*u22          ) *
                            (hn1*v11 + hn2*v12 + hn3*v13))/q;

  theJacobian(3,2) = (sint*(u11*u21 + u12*u22          ) +
                     omcost*(hu1*u21 + hu2*u22          ) +
                     tmsint*(hn1*u21 + hn2*u22          ) *
                            (hn1*u11 + hn2*u12          ))*cosl0/q;

  theJacobian(3,3) = (u11*u21 + u12*u22          );
  
  theJacobian(3,4) = (v11*u21 + v12*u22          );

  //   zt

  theJacobian(4,1) = (sint*(v11*v21 + v12*v22 + v13*v23) +
                     omcost*(hv1*v21 + hv2*v22 + hv3*v23) +
                     tmsint*(hn1*v21 + hn2*v22 + hn3*v23) *
                            (hn1*v11 + hn2*v12 + hn3*v13))/q;

  theJacobian(4,2) = (sint*(u11*v21 + u12*v22          ) +
                     omcost*(hu1*v21 + hu2*v22 + hu3*v23) +
                     tmsint*(hn1*v21 + hn2*v22 + hn3*v23) *
		            (hn1*u11 + hn2*u12          ))*cosl0/q;

  theJacobian(4,3) = (u11*v21 + u12*v22          );

  theJacobian(4,4) = (v11*v21 + v12*v22 + v13*v23);
  // end of TRPRFN
}

#endif

void AnalyticalCurvilinearJacobian::computeInfinitesimalJacobian 
(const GlobalTrajectoryParameters& globalParameters,
 const GlobalPoint&, 
 const GlobalVector& p, 
 const GlobalVector& h, 
 const double& s) {
  /*
   * origin  TRPROP
   *
   C *** ERROR PROPAGATION ALONG A PARTICLE TRAJECTORY IN A MAGNETIC FIELD
   C     ROUTINE ASSUMES THAT IN THE INTERVAL (X1,X2) THE QUANTITIES 1/P
   C     AND (HX,HY,HZ) ARE RATHER CONSTANT. DELTA(PHI) MUST NOT BE TOO LARGE
   C
   C     Authors: A. Haas and W. Wittek
   C
   
  */
  
  
  double qbp = globalParameters.signedInverseMomentum();
  double absS = s;
  
  // average momentum
  GlobalVector tn = (globalParameters.momentum()+p).unit(); 
  double sinl = tn.z(); 
  double cosl = std::sqrt(1.-sinl*sinl); 
  double cosl1 = 1./cosl;
  double tgl=sinl*cosl1;
  double sinp = tn.y()*cosl1;
  double cosp = tn.x()*cosl1;

  // define average magnetic field and gradient 
  // at initial point - inlike TRPROP
  double b0= h.x()*cosp+h.y()*sinp;
  double b2=-h.x()*sinp+h.y()*cosp;
  double b3=-b0*sinl+h.z()*cosl;

  theJacobian(3,2)=absS*cosl;
  theJacobian(4,1)=absS;


  theJacobian(1,0) =  absS*b2;
  //if ( qbp<0) theJacobian(1,0) = -theJacobian(1,0);
  theJacobian(1,2) = -b0*(absS*qbp);
  theJacobian(1,3) =  b3*(b2*qbp*(absS*qbp));
  theJacobian(1,4) = -b2*(b2*qbp*(absS*qbp));
  
  theJacobian(2,0) = -absS*b3*cosl1;
  // if ( qbp<0) theJacobian(2,0) = -theJacobian(2,0);
  theJacobian(2,1) = b0*(absS*qbp)*cosl1*cosl1;
  theJacobian(2,2) = 1.+tgl*b2*(absS*qbp);
  theJacobian(2,3) = -b3*(b3*qbp*(absS*qbp)*cosl1);
  theJacobian(2,4) =  b2*(b3*qbp*(absS*qbp)*cosl1);
  
  theJacobian(3,4) = -b3*tgl*(absS*qbp);
  theJacobian(4,3) =  b3*tgl*(absS*qbp);


}

void
AnalyticalCurvilinearJacobian::computeStraightLineJacobian
(const GlobalTrajectoryParameters& globalParameters,
 const GlobalPoint& x, const GlobalVector& p, const double& s)
{
  //
  // matrix: elements =1 on diagonal and =0 are already set
  // in initialisation
  //
  GlobalVector p1 = globalParameters.momentum().unit();
  double cosl0 = p1.perp();
  theJacobian(3,2) = cosl0 * s;
  theJacobian(4,1) = s;
}
