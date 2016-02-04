#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include <cmath>

PerigeeTrajectoryParameters PerigeeConversions::ftsToPerigeeParameters
  (const FTS& originalFTS, const GlobalPoint& referencePoint, double & pt) const

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

// PerigeeTrajectoryParameters PerigeeConversions::helixToPerigeeParameters
//   (const reco::helix::Parameters & helixPar, const GlobalPoint& referencePoint) const
// {
//   AlgebraicVector theTrackParameters = AlgebraicVector(5);
//   double field  = TrackingTools::FakeField::Field::inTesla(helixPar.vertex()).z() * 2.99792458e-3;
//   theTrackParameters[0] = - field*helixPar.omega();
//   theTrackParameters[1] = atan(1/helixPar.tanDip());
//   theTrackParameters[2] = helixPar.phi0() - M_PI/2;
//   theTrackParameters[3] = helixPar.d0();
//   theTrackParameters[4] = helixPar.dz();
//   return PerigeeTrajectoryParameters(theTrackParameters, helixPar.pt(), true);
// }

PerigeeTrajectoryError PerigeeConversions::ftsToPerigeeError
  (const FTS& originalFTS) const
{
  AlgebraicSymMatrix55 errorMatrix = originalFTS.curvilinearError().matrix();
  AlgebraicMatrix55 curv2perigee = jacobianCurvilinear2Perigee(originalFTS);
  return PerigeeTrajectoryError(ROOT::Math::Similarity(curv2perigee,errorMatrix));
}

// PerigeeTrajectoryError PerigeeConversions::helixToPerigeeError
//   (const reco::helix::Parameters & helixPar, 
// 	const reco::helix::Covariance & helixCov) const
// {
// //FIXME: verify that the order of the parameters are correct
//   AlgebraicSymMatrix55 helixCovMatrix;
//   helixCovMatrix(0,0) = helixCov(reco::helix::i_d0,reco::helix::i_d0);
//   helixCovMatrix(1,1) = helixCov(reco::helix::i_phi0,reco::helix::i_phi0);
//   helixCovMatrix(2,2) = helixCov(reco::helix::i_omega,reco::helix::i_omega);
//   helixCovMatrix(3,3) = helixCov(reco::helix::i_dz,reco::helix::i_dz);
//   helixCovMatrix(4,4) = helixCov(reco::helix::i_tanDip,reco::helix::i_tanDip);
// 
//   helixCovMatrix(0,1) = helixCov(reco::helix::i_d0,reco::helix::i_phi0);
//   helixCovMatrix(0,2) = helixCov(reco::helix::i_d0,reco::helix::i_omega);
//   helixCovMatrix(0,3) = helixCov(reco::helix::i_d0,reco::helix::i_dz);
//   helixCovMatrix(0,4) = helixCov(reco::helix::i_d0,reco::helix::i_tanDip);
// 
//   helixCovMatrix(1,2) = helixCov(reco::helix::i_phi0,reco::helix::i_omega);
//   helixCovMatrix(1,3) = helixCov(reco::helix::i_phi0,reco::helix::i_dz);
//   helixCovMatrix(1,4) = helixCov(reco::helix::i_phi0,reco::helix::i_tanDip);
// 
//   helixCovMatrix(2,3) = helixCov(reco::helix::i_omega,reco::helix::i_dz);
//   helixCovMatrix(2,4) = helixCov(reco::helix::i_omega,reco::helix::i_tanDip);
// 
//   helixCovMatrix(3,4) = helixCov(reco::helix::i_dz,reco::helix::i_tanDip);
// 
//   AlgebraicMatrix helix2perigee = jacobianHelix2Perigee(helixPar, helixCov);
//   return PerigeeTrajectoryError(helixCovMatrix.similarity(helix2perigee));
// }


CurvilinearTrajectoryError PerigeeConversions::curvilinearError
  (const PerigeeTrajectoryError& perigeeError, const GlobalTrajectoryParameters& gtp) const
{
  AlgebraicMatrix55 perigee2curv = jacobianPerigee2Curvilinear(gtp);
  return CurvilinearTrajectoryError(ROOT::Math::Similarity(perigee2curv, perigeeError.covarianceMatrix()));
}

GlobalPoint PerigeeConversions::positionFromPerigee
  (const PerigeeTrajectoryParameters& parameters, const GlobalPoint& referencePoint) const
{
  AlgebraicVector5 theVector = parameters.vector();
  return GlobalPoint(theVector[3]*sin(theVector[2])+referencePoint.x(),
  		     -theVector[3]*cos(theVector[2])+referencePoint.y(),
		     theVector[4]+referencePoint.z());
}


GlobalVector PerigeeConversions::momentumFromPerigee
  (const PerigeeTrajectoryParameters& parameters, double pt, const GlobalPoint& referencePoint) const
{
  return GlobalVector(cos(parameters.phi()) * pt,
  		      sin(parameters.phi()) * pt,
   		      pt / tan(parameters.theta()));
}

GlobalVector PerigeeConversions::momentumFromPerigee
  (const AlgebraicVector& momentum, const TrackCharge& charge, 
   const GlobalPoint& referencePoint, const MagneticField* field) const {
      return momentumFromPerigee(asSVector<3>(momentum), charge, referencePoint, field);
  }

GlobalVector PerigeeConversions::momentumFromPerigee
  (const AlgebraicVector3& momentum, const TrackCharge& charge, 
   const GlobalPoint& referencePoint, const MagneticField* field) const
{
  double pt;
  if (momentum[0]==0.) throw cms::Exception("PerigeeConversions", "Track with rho=0");

  double bz = fabs(field->inInverseGeV(referencePoint).z());
  if ( charge!=0 && bz>1.e-10 ) {
    pt = std::abs(bz/momentum[0]);
    if (pt<1.e-10) throw cms::Exception("PerigeeConversions", "pt is 0");
  } else {
    pt = 1 / momentum[0];
  }

  return GlobalVector(cos(momentum[2]) * pt,
  		      sin(momentum[2]) * pt,
   		      pt/tan(momentum[1]));
}

TrackCharge PerigeeConversions::chargeFromPerigee
  (const PerigeeTrajectoryParameters& parameters) const
{
  return parameters.charge();
}

TrajectoryStateClosestToPoint PerigeeConversions::trajectoryStateClosestToPoint
	(const AlgebraicVector& momentum, const GlobalPoint& referencePoint,
	 const TrackCharge& charge, const AlgebraicMatrix& theCovarianceMatrix, ///FIXME !!! why not Sym !!??
	 const MagneticField* field) const {
            AlgebraicSymMatrix sym; sym.assign(theCovarianceMatrix); // below, this was used for Matrix => SymMatrix
            return trajectoryStateClosestToPoint(asSVector<3>(momentum), referencePoint, 
                    charge, asSMatrix<6>(sym), field);

        }


TrajectoryStateClosestToPoint PerigeeConversions::trajectoryStateClosestToPoint
	(const AlgebraicVector3& momentum, const GlobalPoint& referencePoint,
	 const TrackCharge& charge, const AlgebraicSymMatrix66& theCovarianceMatrix,
	 const MagneticField* field) const
{
  AlgebraicMatrix66 param2cart = jacobianParameters2Cartesian
  	(momentum, referencePoint, charge, field);
  CartesianTrajectoryError cartesianTrajErr(ROOT::Math::Similarity(param2cart, theCovarianceMatrix));

  FTS theFTS(GlobalTrajectoryParameters(referencePoint,
	    momentumFromPerigee(momentum, charge, referencePoint, field), charge,
	    field), cartesianTrajErr);

  return TrajectoryStateClosestToPoint(theFTS, referencePoint);
}

AlgebraicMatrix
PerigeeConversions::jacobianParameters2Cartesian_old(
	const AlgebraicVector& momentum, const GlobalPoint& position,
	const TrackCharge& charge, const MagneticField* field) const {
    return asHepMatrix(jacobianParameters2Cartesian(asSVector<3>(momentum), position, charge, field));
}

AlgebraicMatrix66
PerigeeConversions::jacobianParameters2Cartesian(
	const AlgebraicVector3& momentum, const GlobalPoint& position,
	const TrackCharge& charge, const MagneticField* field) const
{
  if (momentum[0]==0.) throw cms::Exception("PerigeeConversions", "Track with rho=0");
  double factor = 1.;
  double bField = field->inInverseGeV(position).z();
  if (charge!=0 && fabs(bField)>1.e-10) {
//     if (bField==0.) throw cms::Exception("PerigeeConversions", "Field is 0");
    factor = -bField*charge;
  }
  AlgebraicMatrix66 frameTransJ;
  frameTransJ(0,0) = 1;
  frameTransJ(1,1) = 1;
  frameTransJ(2,2) = 1;
  frameTransJ(3,3) = - factor * cos(momentum[2]) / (momentum[0]*momentum[0]);
  frameTransJ(4,3) = - factor * sin(momentum[2]) / (momentum[0]*momentum[0]);
  frameTransJ(5,3) = - factor / tan(momentum[1]) / (momentum[0]*momentum[0]);

  frameTransJ(3,5) = - factor * sin(momentum[2])  / (momentum[0]);
  frameTransJ(4,5) = factor * cos(momentum[2]) / (momentum[0]);
  frameTransJ(5,4) = - factor / (momentum[0]*sin(momentum[1])*sin(momentum[1]));

  return frameTransJ;
}

AlgebraicMatrix
PerigeeConversions::jacobianCurvilinear2Perigee_old(const FreeTrajectoryState& fts) const {
    return asHepMatrix(jacobianCurvilinear2Perigee(fts));
}


AlgebraicMatrix55
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
  GlobalVector B  = fts.parameters().magneticField().inInverseGeV(x);
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

  AlgebraicMatrix55 jac;

  if( fabs(fts.transverseCurvature())<1.e-10 ) {
    jac(0,0) = 1/coslambda;
    jac(0,1) = tanlambda/coslambda/fts.parameters().momentum().mag();
  }else{
    double Bz = B.z();
    jac(0,0) = -Bz/coslambda;
    jac(0,1) = -Bz * tanlambda/coslambda*qbp;
    jac(1,3) = alphaQ * NV * UI/TI;
    jac(1,4) = alphaQ * NV * VI/TI;
    jac(0,3) = -jac(0,1) * jac(1,3);
    jac(0,4) = -jac(0,1) * jac(1,4);
    jac(2,3) = -alphaQ/coslambda * NU * UI/TI;
    jac(2,4) = -alphaQ/coslambda * NU * VI/TI;
  }
  jac(1,1) = -1.;
  jac(2,2) = 1.;
  jac(3,3) = VK/TI;
  jac(3,4) = -UK/TI;
  jac(4,3) = -VJ/TI;
  jac(4,4) = UJ/TI;
  
  return jac;
  
}


AlgebraicMatrix 
PerigeeConversions::jacobianPerigee2Curvilinear_old(const GlobalTrajectoryParameters& gtp) const {
    return asHepMatrix(jacobianPerigee2Curvilinear(gtp));
}

AlgebraicMatrix55 
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
  GlobalVector B  = gtp.magneticField().inInverseGeV(x);
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

  AlgebraicMatrix55 jac;

  if( fabs(gtp.transverseCurvature())<1.e-10 ) {
    jac(0,0) = coslambda;
    jac(0,1) = sinlambda/coslambda/gtp.momentum().mag();
  }else{
    jac(0,0) = -coslambda/B.z();
    jac(0,1) = -sinlambda * mqbpt;
    jac(1,3) = -alphaQ * NV * TJ;
    jac(1,4) = -alphaQ * NV * TK;
    jac(2,3) = -alphaQ/coslambda * NU * TJ;
    jac(2,4) = -alphaQ/coslambda * NU * TK;
  }
  jac(1,1) = -1.;
  jac(2,2) = 1.;
  jac(3,3) = UJ;
  jac(3,4) = UK;
  jac(4,3) = VJ;
  jac(4,4) = VK;
  
  return jac;
}

// AlgebraicMatrix 
// PerigeeConversions::jacobianHelix2Perigee(const reco::helix::Parameters & helixPar, 
// 	const reco::helix::Covariance & helixCov) const
// {
//   AlgebraicMatrix55 jac;
// 
//   jac(3,0) = 1.;
//   jac(2,1) = 1.;
// //   jac(0,2) = - 1. / magField.inTesla(helixPar.vertex()).z() * 2.99792458e-3;
//   jac(0,2) = - 1. / (TrackingTools::FakeField::Field::inTesla(helixPar.vertex()).z() * 2.99792458e-3);
//   jac(4,3) = 1.;
//   jac(1,4) = -(1. + helixPar.tanDip()*helixPar.tanDip());
// std::std::cout << jac;
//   return jac;
// }
