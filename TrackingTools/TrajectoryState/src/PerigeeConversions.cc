#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include <cmath>
#include <vdt/vdtMath.h>

PerigeeTrajectoryParameters PerigeeConversions::ftsToPerigeeParameters
  (const FTS& originalFTS, const GlobalPoint& referencePoint, double & pt)

{
  GlobalVector impactDistance = originalFTS.position() - referencePoint;

  pt = originalFTS.momentum().perp();
  if (pt==0.) throw cms::Exception("PerigeeConversions", "Track with pt=0");
  
  double theta = originalFTS.momentum().theta();
  double phi = originalFTS.momentum().phi();
  double field  = originalFTS.parameters().magneticField().inInverseGeV(originalFTS.position()).z();
//   if (field==0.) throw cms::Exception("PerigeeConversions", "Field is 0") << " at " << originalFTS.position() << "\n" ;

  double positiveMomentumPhi = ( (phi>0) ? phi : (2*M_PI+phi) );
  double positionPhi = impactDistance.phi();
  double positivePositionPhi =
   ( (positionPhi>0) ? positionPhi : (2*M_PI+positionPhi) );
  double phiDiff = positiveMomentumPhi - positivePositionPhi;
  if (phiDiff<0.0) phiDiff+= (2*M_PI);
  double signEpsilon = ( (phiDiff > M_PI) ? -1.0 : 1.0);

  double epsilon = signEpsilon *
  		     sqrt ( impactDistance.x()*impactDistance.x() +
   			    impactDistance.y()*impactDistance.y() );

  // The track parameters:
  AlgebraicVector5 theTrackParameters;

  double signTC = - originalFTS.charge();
  bool isCharged = (signTC!=0) && (fabs(field)>1.e-10);
  if (isCharged) {
    theTrackParameters[0] = field / pt*signTC;
  } else {
    theTrackParameters[0] = 1 / pt;
  }
  theTrackParameters[1] = theta;
  theTrackParameters[2] = phi;
  theTrackParameters[3] = epsilon;
  theTrackParameters[4] = impactDistance.z();
  return PerigeeTrajectoryParameters(theTrackParameters, isCharged);
}


PerigeeTrajectoryError PerigeeConversions::ftsToPerigeeError
  (const FTS& originalFTS)
{
  AlgebraicSymMatrix55 errorMatrix = originalFTS.curvilinearError().matrix();
  AlgebraicMatrix55 curv2perigee = jacobianCurvilinear2Perigee(originalFTS);
  return PerigeeTrajectoryError(ROOT::Math::Similarity(curv2perigee,errorMatrix));
}


CurvilinearTrajectoryError PerigeeConversions::curvilinearError
  (const PerigeeTrajectoryError& perigeeError, const GlobalTrajectoryParameters& gtp)
{
  AlgebraicMatrix55 perigee2curv = jacobianPerigee2Curvilinear(gtp);
  return CurvilinearTrajectoryError(ROOT::Math::Similarity(perigee2curv, perigeeError.covarianceMatrix()));
}

GlobalPoint PerigeeConversions::positionFromPerigee
  (const PerigeeTrajectoryParameters& parameters, const GlobalPoint& referencePoint)
{
  AlgebraicVector5 theVector = parameters.vector();
  return GlobalPoint(theVector[3]*sin(theVector[2])+referencePoint.x(),
  		     -theVector[3]*cos(theVector[2])+referencePoint.y(),
		     theVector[4]+referencePoint.z());
}


GlobalVector PerigeeConversions::momentumFromPerigee
  (const PerigeeTrajectoryParameters& parameters, double pt, const GlobalPoint& referencePoint)
{
  return GlobalVector(vdt::fast_cos(parameters.phi()) * pt,
  		      vdt::fast_sin(parameters.phi()) * pt,
   		      pt /vdt::fast_tan(parameters.theta()));
}


GlobalVector PerigeeConversions::momentumFromPerigee
  (const AlgebraicVector3& momentum, const TrackCharge& charge, 
   const GlobalPoint& referencePoint, const MagneticField* field)
{
  double pt;

  double bz = fabs(field->inInverseGeV(referencePoint).z());
  if ( charge!=0 && bz>1.e-10 ) {
    pt = std::abs(bz/momentum[0]);
    // if (pt<1.e-10) throw cms::Exception("PerigeeConversions", "pt is 0");
  } else {
    pt = 1 / momentum[0];
  }

  return GlobalVector(vdt::fast_cos(momentum[2]) * pt,
  		      vdt::fast_sin(momentum[2]) * pt,
   		      pt/vdt::fast_tan(momentum[1]));
}


TrajectoryStateClosestToPoint PerigeeConversions::trajectoryStateClosestToPoint
	(const AlgebraicVector3& momentum, const GlobalPoint& referencePoint,
	 const TrackCharge& charge, const AlgebraicSymMatrix66& theCovarianceMatrix,
	 const MagneticField* field)
{
  AlgebraicMatrix66 param2cart = jacobianParameters2Cartesian
  	(momentum, referencePoint, charge, field);
  CartesianTrajectoryError cartesianTrajErr(ROOT::Math::Similarity(param2cart, theCovarianceMatrix));

  FTS theFTS(GlobalTrajectoryParameters(referencePoint,
	    momentumFromPerigee(momentum, charge, referencePoint, field), charge,
	    field), cartesianTrajErr);

  return TrajectoryStateClosestToPoint(theFTS, referencePoint);
}


AlgebraicMatrix66
PerigeeConversions::jacobianParameters2Cartesian(
	const AlgebraicVector3& momentum, const GlobalPoint& position,
	const TrackCharge& charge, const MagneticField* field)
{
  float factor = -1.;
  float bField = field->inInverseGeV(position).z();
  if (charge!=0 && std::abs(bField)>1.e-10f)
    factor = bField*charge;
 

  float s1,c1; vdt::fast_sincosf(momentum[1],s1,c1);
  float s2,c2; vdt::fast_sincosf(momentum[2],s2,c2);
  float f1 = factor/(momentum[0]*momentum[0]);
  float f2 = factor/momentum[0];

  AlgebraicMatrix66 frameTransJ;
  frameTransJ(0,0) = 1;
  frameTransJ(1,1) = 1;
  frameTransJ(2,2) = 1;
  frameTransJ(3,3) =  (f1 * c2); 
  frameTransJ(4,3) =  (f1 * s2);
  frameTransJ(5,3) =  (f1*c1/s1);  

  frameTransJ(3,5) = (f2 * s2);
  frameTransJ(4,5) = -(f2 * c2);
  frameTransJ(5,4) = (f2/(s1*s1));

  return frameTransJ;
}


AlgebraicMatrix55
PerigeeConversions::jacobianCurvilinear2Perigee(const FreeTrajectoryState& fts){

  GlobalVector p = fts.momentum();

  GlobalVector Z = GlobalVector(0.,0.,1.);
  GlobalVector T = p.unit();
  GlobalVector U = Z.cross(T).unit();; 
  GlobalVector V = T.cross(U);

  GlobalVector I = GlobalVector(-p.x(), -p.y(), 0.); //opposite to track dir.
  I = I.unit();
  GlobalVector J(-I.y(), I.x(),0.); //counterclockwise rotation
  GlobalVector K(Z);
  GlobalPoint x = fts.position();
  GlobalVector B  = fts.parameters().magneticField().inInverseGeV(x);
  GlobalVector H = B.unit();
  GlobalVector HxT = H.cross(T);
  GlobalVector N = HxT.unit();
  double alpha = HxT.mag();
  double qbp = fts.signedInverseMomentum();
  double Q = -B.mag() * qbp;
  double alphaQ = alpha * Q;

  double lambda = 0.5 * M_PI - p.theta();
  double sinlambda, coslambda;  vdt::fast_sincos(lambda, sinlambda, coslambda);
  double seclambda = 1./coslambda;

  double ITI = 1./T.dot(I);
  double NU = N.dot(U);
  double NV = N.dot(V);
  double UI = U.dot(I);
  double VI = V.dot(I);
  double UJ = U.dot(J);
  double VJ = V.dot(J);
  double UK = U.dot(K);
  double VK = V.dot(K);

  AlgebraicMatrix55 jac;

  if( fabs(fts.transverseCurvature())<1.e-10 ) {
    jac(0,0) = seclambda;
    jac(0,1) = sinlambda*seclambda*seclambda*std::abs(qbp);
  }else{
    double Bz = B.z();
    jac(0,0) = -Bz * seclambda;
    jac(0,1) = -Bz * sinlambda*seclambda*seclambda*qbp;
    jac(1,3) = alphaQ * NV * UI*ITI;
    jac(1,4) = alphaQ * NV * VI*ITI;
    jac(0,3) = -jac(0,1) * jac(1,3);
    jac(0,4) = -jac(0,1) * jac(1,4);
    jac(2,3) = -alphaQ*seclambda * NU * UI*ITI;
    jac(2,4) = -alphaQ*seclambda * NU * VI*ITI;
  }
  jac(1,1) = -1.;
  jac(2,2) = 1.;
  jac(3,3) = VK*ITI;
  jac(3,4) = -UK*ITI;
  jac(4,3) = -VJ*ITI;
  jac(4,4) = UJ*ITI;
  
  return jac;
  
}



AlgebraicMatrix55 
PerigeeConversions::jacobianPerigee2Curvilinear(const GlobalTrajectoryParameters& gtp) {

  GlobalVector p = gtp.momentum();

  GlobalVector Z = GlobalVector(0.f,0.f,1.f);
  GlobalVector T = p.unit();
  GlobalVector U = Z.cross(T).unit();; 
  GlobalVector V = T.cross(U);

  GlobalVector I = GlobalVector(-p.x(), -p.y(), 0.f); //opposite to track dir.
  I = I.unit();
  GlobalVector J(-I.y(), I.x(),0.f); //counterclockwise rotation
  GlobalVector K(Z);
  GlobalPoint x = gtp.position();
  GlobalVector B  = gtp.magneticField().inInverseGeV(x);
  GlobalVector H = B.unit();
  GlobalVector HxT = H.cross(T);
  GlobalVector N = HxT.unit();
  double alpha = HxT.mag();
  double qbp = gtp.signedInverseMomentum();
  double Q = -B.mag() * qbp;
  double alphaQ = alpha * Q;


  double lambda = 0.5 * M_PI - p.theta();
  double sinlambda, coslambda;  vdt::fast_sincos(lambda, sinlambda, coslambda);
  double seclambda = 1./coslambda;

  double mqbpt = -1./coslambda * qbp;

  double TJ = T.dot(J);
  double TK = T.dot(K);
  double NU = N.dot(U);
  double NV = N.dot(V);
  double UJ = U.dot(J);
  double VJ = V.dot(J);
  double UK = U.dot(K);
  double VK = V.dot(K);

  AlgebraicMatrix55 jac;

  if( fabs(gtp.transverseCurvature())<1.e-10f ) {
    jac(0,0) = coslambda;
    jac(0,1) = sinlambda/coslambda/gtp.momentum().mag();
  }else{
    jac(0,0) = -coslambda/B.z();
    jac(0,1) = -sinlambda * mqbpt;
    jac(1,3) = -alphaQ * NV * TJ;
    jac(1,4) = -alphaQ * NV * TK;
    jac(2,3) = -alphaQ*seclambda * NU * TJ;
    jac(2,4) = -alphaQ*seclambda * NU * TK;
  }
  jac(1,1) = -1.;
  jac(2,2) = 1.;
  jac(3,3) = UJ;
  jac(3,4) = UK;
  jac(4,3) = VJ;
  jac(4,4) = VK;
  
  return jac;
}

