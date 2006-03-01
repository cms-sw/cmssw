#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "TrackingTools/TrajectoryState/interface/FakeField.h"
#include <cmath>

PerigeeTrajectoryParameters PerigeeConversions::ftsToPerigeeParameters
  (const FTS& originalFTS, const GlobalPoint& referencePoint) const

{
  GlobalVector impactDistance = originalFTS.position() - referencePoint;

  double theta = originalFTS.momentum().theta();
  double phi = originalFTS.momentum().phi();
  double pt = originalFTS.momentum().perp();
//   double field = MagneticField::inInverseGeV(originalFTS.position()).z();
  double field  = originalFTS.parameters().magneticField().inTesla(originalFTS.position()).z() * 2.99792458e-3;

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
  AlgebraicVector theTrackParameters = AlgebraicVector(5);

  double signTC = -originalFTS.charge();
  bool isCharged = (signTC!=0);
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

PerigeeTrajectoryParameters PerigeeConversions::helixToPerigeeParameters
  (const reco::helix::Parameters & helixPar, const GlobalPoint& referencePoint) const
{
  AlgebraicVector theTrackParameters = AlgebraicVector(5);
  double field  = TrackingTools::FakeField::Field::inTesla(helixPar.vertex()).z() * 2.99792458e-3;
  theTrackParameters[0] = - field*helixPar.omega();
  theTrackParameters[1] = atan(1/helixPar.tanDip());
  theTrackParameters[2] = helixPar.phi0() - M_PI/2;
  theTrackParameters[3] = helixPar.d0();
  theTrackParameters[4] = helixPar.dz();
  return PerigeeTrajectoryParameters(theTrackParameters, true);
}

PerigeeTrajectoryError PerigeeConversions::ftsToPerigeeError
  (const FTS& originalFTS) const
{
  AlgebraicSymMatrix errorMatrix = originalFTS.curvilinearError().matrix();
  AlgebraicMatrix curv2perigee = jacobianCurvilinear2Perigee(originalFTS);
  return PerigeeTrajectoryError(errorMatrix.similarity(curv2perigee));
}

PerigeeTrajectoryError PerigeeConversions::helixToPerigeeError
  (const reco::helix::Parameters & helixPar, 
	const reco::helix::Covariance & helixCov) const
{
//FIXME: verify that the order of the parameters are correct
  AlgebraicSymMatrix helixCovMatrix(5,0);
  helixCovMatrix(1,1) = helixCov(reco::helix::i_d0,reco::helix::i_d0);
  helixCovMatrix(2,2) = helixCov(reco::helix::i_phi0,reco::helix::i_phi0);
  helixCovMatrix(3,3) = helixCov(reco::helix::i_omega,reco::helix::i_omega);
  helixCovMatrix(4,4) = helixCov(reco::helix::i_dz,reco::helix::i_dz);
  helixCovMatrix(5,5) = helixCov(reco::helix::i_tanDip,reco::helix::i_tanDip);

  helixCovMatrix(1,2) = helixCov(reco::helix::i_d0,reco::helix::i_phi0);
  helixCovMatrix(1,3) = helixCov(reco::helix::i_d0,reco::helix::i_omega);
  helixCovMatrix(1,4) = helixCov(reco::helix::i_d0,reco::helix::i_dz);
  helixCovMatrix(1,5) = helixCov(reco::helix::i_d0,reco::helix::i_tanDip);

  helixCovMatrix(2,3) = helixCov(reco::helix::i_phi0,reco::helix::i_omega);
  helixCovMatrix(2,4) = helixCov(reco::helix::i_phi0,reco::helix::i_dz);
  helixCovMatrix(2,5) = helixCov(reco::helix::i_phi0,reco::helix::i_tanDip);

  helixCovMatrix(3,4) = helixCov(reco::helix::i_omega,reco::helix::i_dz);
  helixCovMatrix(3,5) = helixCov(reco::helix::i_omega,reco::helix::i_tanDip);

  helixCovMatrix(5,5) = helixCov(reco::helix::i_dz,reco::helix::i_tanDip);

  AlgebraicMatrix helix2perigee = jacobianHelix2Perigee(helixPar, helixCov);
  return PerigeeTrajectoryError(helixCovMatrix.similarity(helix2perigee));
}


CurvilinearTrajectoryError PerigeeConversions::curvilinearError
  (const PerigeeTrajectoryError& perigeeError, const GlobalTrajectoryParameters& gtp) const
{
  AlgebraicMatrix perigee2curv = jacobianPerigee2Curvilinear(gtp);
  return CurvilinearTrajectoryError(perigeeError.covarianceMatrix().similarity(perigee2curv));
}

GlobalPoint PerigeeConversions::positionFromPerigee
  (const PerigeeTrajectoryParameters& parameters, const GlobalPoint& referencePoint) const
{
  AlgebraicVector theVector = parameters.vector();
  return GlobalPoint(theVector[3]*sin(theVector[2])+referencePoint.x(),
  		     -theVector[3]*cos(theVector[2])+referencePoint.y(),
		     theVector[4]+referencePoint.z());
}


GlobalVector PerigeeConversions::momentumFromPerigee
  (const PerigeeTrajectoryParameters& parameters, const GlobalPoint& referencePoint) const
{
  return momentumFromPerigee(parameters.vector(), parameters.charge(), referencePoint);
}

GlobalVector PerigeeConversions::momentumFromPerigee
  (const AlgebraicVector& momentum, const TrackCharge& charge, 
   const GlobalPoint& referencePoint) const
{
  double pt;
  if (charge!=0) {
//     pt = abs(MagneticField::inInverseGeV(referencePoint).z() / momentum[0]);
//     pt = std::abs(magField.inTesla(referencePoint).z() * 2.99792458e-3 / momentum[0]);
     pt = std::abs(TrackingTools::FakeField::Field::inTesla(referencePoint).z() * 2.99792458e-3 / momentum[0]);
  } else {
    pt = 1 / momentum[0];
  }
  return GlobalVector(cos(momentum[2]) * pt,
  		      sin(momentum[2]) * pt,
   		      pt/tan(momentum[1]));
}

TrackCharge PerigeeConversions::chargeFromPerigee
  (const PerigeeTrajectoryParameters& parameters, const GlobalPoint& referencePoint) const
{
  return parameters.charge();
}


TrajectoryStateClosestToPoint PerigeeConversions::trajectoryStateClosestToPoint
	(const AlgebraicVector& momentum, const GlobalPoint& referencePoint,
	 const TrackCharge& charge, const AlgebraicMatrix& theCovarianceMatrix) const
{
  AlgebraicMatrix param2cart = jacobianParameters2Cartesian
  	(momentum, referencePoint, charge);
  AlgebraicSymMatrix cartesianErrorMatrix(6,0);
  cartesianErrorMatrix.assign(param2cart*theCovarianceMatrix*param2cart.T());
  CartesianTrajectoryError cartesianTrajErr(cartesianErrorMatrix);

  FTS theFTS(GlobalTrajectoryParameters(referencePoint,
	    momentumFromPerigee(momentum, charge, referencePoint), charge,
	    TrackingTools::FakeField::Field::field()), cartesianTrajErr);

  return TrajectoryStateClosestToPoint(theFTS, referencePoint);
}


AlgebraicMatrix
PerigeeConversions::jacobianParameters2Cartesian(
	const AlgebraicVector& momentum, const GlobalPoint& position,
	const TrackCharge& charge) const
{
  double factor = 1.;
  if (charge!=0) {
    double field = TrackingTools::FakeField::Field::inTesla(position).z() * 2.99792458e-3;
//     MagneticField::inInverseGeV(position).z();
    factor = -field*charge;
  }
  AlgebraicMatrix frameTransJ(6, 6, 0);
  frameTransJ[0][0] = 1;
  frameTransJ[1][1] = 1;
  frameTransJ[2][2] = 1;
  frameTransJ[3][3] = - factor * cos(momentum[2]) / (momentum[0]*momentum[0]);
  frameTransJ[4][3] = - factor * sin(momentum[2]) / (momentum[0]*momentum[0]);
  frameTransJ[5][3] = - factor / tan(momentum[1]) / (momentum[0]*momentum[0]);

  frameTransJ[3][5] = - factor * sin(momentum[2])  / (momentum[0]);
  frameTransJ[4][5] = factor * cos(momentum[2]) / (momentum[0]);
  frameTransJ[5][4] = - factor / (momentum[0]*sin(momentum[1])*sin(momentum[1]));

  return frameTransJ;
}

AlgebraicMatrix 
PerigeeConversions::jacobianCurvilinear2Perigee(const FreeTrajectoryState& fts) const {

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
//   GlobalVector B = MagneticField::inInverseGeV(x);
  GlobalVector B  = fts.parameters().magneticField().inTesla(x) * 2.99792458e-3;
  GlobalVector H = B.unit();
  GlobalVector HxT = H.cross(T);
  GlobalVector N = HxT.unit();
  double alpha = HxT.mag();
  double qbp = fts.signedInverseMomentum();
  double Q = -B.mag() * qbp;
  double alphaQ = alpha * Q;

  double lambda = 0.5 * M_PI - p.theta();
  double coslambda = cos(lambda), tanlambda = tan(lambda);

  double TI = T.dot(I);
  double NU = N.dot(U);
  double NV = N.dot(V);
  double UI = U.dot(I);
  double VI = V.dot(I);
  double UJ = U.dot(J);
  double VJ = V.dot(J);
  double UK = U.dot(K);
  double VK = V.dot(K);

  AlgebraicMatrix jac(5,5,0);

  if( fabs(fts.transverseCurvature())<1.e-10 ) {
    jac(1,1) = 1/coslambda;
    jac(1,2) = tanlambda/coslambda/fts.parameters().momentum().mag();
  }else{
    double Bz = B.z();
    jac(1,1) = -Bz/coslambda;
    jac(1,2) = -Bz * tanlambda/coslambda*qbp;
    jac(2,4) = alphaQ * NV * UI/TI;
    jac(2,5) = alphaQ * NV * VI/TI;
    jac(1,4) = -jac(1,2) * jac(2,4);
    jac(1,5) = -jac(1,2) * jac(2,5);
    jac(3,4) = -alphaQ/coslambda * NU * UI/TI;
    jac(3,5) = -alphaQ/coslambda * NU * VI/TI;
  }
  jac(2,2) = -1.;
  jac(3,3) = 1.;
  jac(4,4) = VK/TI;
  jac(4,5) = -UK/TI;
  jac(5,4) = -VJ/TI;
  jac(5,5) = UJ/TI;
  
  return jac;
  
}


AlgebraicMatrix 
PerigeeConversions::jacobianPerigee2Curvilinear(const GlobalTrajectoryParameters& gtp) const {

  GlobalVector p = gtp.momentum();

  GlobalVector Z = GlobalVector(0.,0.,1.);
  GlobalVector T = p.unit();
  GlobalVector U = Z.cross(T).unit();; 
  GlobalVector V = T.cross(U);

  GlobalVector I = GlobalVector(-p.x(), -p.y(), 0.); //opposite to track dir.
  I = I.unit();
  GlobalVector J(-I.y(), I.x(),0.); //counterclockwise rotation
  GlobalVector K(Z);
  GlobalPoint x = gtp.position();
//   GlobalVector B = MagneticField::inInverseGeV(x);
  GlobalVector B  = gtp.magneticField().inTesla(x) * 2.99792458e-3;
  GlobalVector H = B.unit();
  GlobalVector HxT = H.cross(T);
  GlobalVector N = HxT.unit();
  double alpha = HxT.mag();
  double qbp = gtp.signedInverseMomentum();
  double Q = -B.mag() * qbp;
  double alphaQ = alpha * Q;

  double lambda = 0.5 * M_PI - p.theta();
  double coslambda = cos(lambda), sinlambda = sin(lambda);
  double mqbpt = -1./coslambda * qbp;

  double TJ = T.dot(J);
  double TK = T.dot(K);
  double NU = N.dot(U);
  double NV = N.dot(V);
  double UJ = U.dot(J);
  double VJ = V.dot(J);
  double UK = U.dot(K);
  double VK = V.dot(K);

  AlgebraicMatrix jac(5,5,0);

  if( fabs(gtp.transverseCurvature())<1.e-10 ) {
    jac(1,1) = coslambda;
    jac(1,2) = sinlambda/coslambda/gtp.momentum().mag();
  }else{
    jac(1,1) = -coslambda/B.z();
    jac(1,2) = -sinlambda * mqbpt;
    jac(2,4) = -alphaQ * NV * TJ;
    jac(2,5) = -alphaQ * NV * TK;
    jac(3,4) = -alphaQ/coslambda * NU * TJ;
    jac(3,5) = -alphaQ/coslambda * NU * TK;
  }
  jac(2,2) = -1.;
  jac(3,3) = 1.;
  jac(4,4) = UJ;
  jac(4,5) = UK;
  jac(5,4) = VJ;
  jac(5,5) = VK;
  
  return jac;
}

AlgebraicMatrix 
PerigeeConversions::jacobianHelix2Perigee(const reco::helix::Parameters & helixPar, 
	const reco::helix::Covariance & helixCov) const
{
  AlgebraicMatrix jac(5,5,0);

  jac(4,1) = 1.;
  jac(3,2) = 1.;
//   jac(1,3) = - 1. / magField.inTesla(helixPar.vertex()).z() * 2.99792458e-3;
  jac(1,3) = - 1. / TrackingTools::FakeField::Field::inTesla(helixPar.vertex()).z() * 2.99792458e-3;
  jac(5,4) = 1.;
  jac(2,5) = -(1. + helixPar.tanDip()*helixPar.tanDip());

  return jac;
}
