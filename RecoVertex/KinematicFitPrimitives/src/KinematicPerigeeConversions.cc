#include "RecoVertex/KinematicFitPrimitives/interface/KinematicPerigeeConversions.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"


ExtendedPerigeeTrajectoryParameters
KinematicPerigeeConversions::extendedPerigeeFromKinematicParameters(
	const KinematicState& state, const GlobalPoint& point) const
{
//making an extended perigee parametrization
//out of kinematic state and point defined
 AlgebraicVector res(6);
 res(6) = state.mass();
 AlgebraicVector par = state.kinematicParameters().vector();
 GlobalVector impactDistance = state.globalPosition() - point;
 double theta = state.globalMomentum().theta();
 double phi = state.globalMomentum().phi();
 double pt  = state.globalMomentum().transverse();
//  double field = MagneticField::inGeVPerCentimeter(state.globalPosition()).z();
  double field  = state.magneticField()->inInverseGeV(state.globalPosition()).z();
// double signTC = -state.particleCharge();
// double transverseCurvature = field/pt*signTC;

//making a proper sign for epsilon
 double positiveMomentumPhi  = ((phi>0) ? phi : (2*M_PI + phi));
 double positionPhi = impactDistance.phi();
 double positivePositionPhi =
   ( (positionPhi>0) ? positionPhi : (2*M_PI+positionPhi) );
 double phiDiff = positiveMomentumPhi - positivePositionPhi;
 if (phiDiff<0.0) phiDiff+= (2*M_PI);
 double signEpsilon = ( (phiDiff > M_PI) ? -1.0 : 1.0);

 double epsilon = signEpsilon *
  		  sqrt(impactDistance.x()*impactDistance.x() +
   		       impactDistance.y()*impactDistance.y());

 double signTC = -state.particleCharge();
 bool isCharged = (signTC!=0);

 if (isCharged) {
   res(1) = field / pt*signTC;
 } else {
   res(1) = 1 / pt;
 }

 res(2) = theta;
 res(3) = phi;
 res(4) = epsilon;
 res(5) = impactDistance.z();
 return ExtendedPerigeeTrajectoryParameters(res,state.particleCharge());
}

KinematicParameters KinematicPerigeeConversions::kinematicParametersFromExPerigee(
	const ExtendedPerigeeTrajectoryParameters& pr, const GlobalPoint& point,
	const MagneticField* field) const
{
 AlgebraicVector par(7);
 AlgebraicVector theVector = pr.vector();
 double pt;
 if(pr.charge() !=0){
  pt = std::abs(field->inInverseGeV(point).z() / theVector(1));
 }else{pt = 1/theVector(1);}
 par(7) = theVector(6);
 par(1) = theVector[3]*sin(theVector[2])+point.x();
 par(2) = -theVector[3]*cos(theVector[2])+point.y();
 par(3) = theVector[4]+point.z();
 par(4) = cos(theVector[2]) * pt;
 par(5) = sin(theVector[2]) * pt;
 par(6) = pt/tan(theVector[1]);

 return KinematicParameters(par);
}


KinematicState
KinematicPerigeeConversions::kinematicState(const AlgebraicVector& momentum,
	const GlobalPoint& referencePoint, const TrackCharge& charge,
	const AlgebraicMatrix& theCovarianceMatrix, const MagneticField* field) const
{
 AlgebraicMatrix param2cart = jacobianParameters2Kinematic(momentum,
				referencePoint, charge, field);
 AlgebraicSymMatrix kinematicErrorMatrix(7,0);
 kinematicErrorMatrix.assign(param2cart*theCovarianceMatrix*param2cart.T());

 KinematicParametersError kinematicParamError(kinematicErrorMatrix);
 AlgebraicVector par(7);
 AlgebraicVector mm = momentumFromPerigee(momentum, referencePoint, charge, field);
 par(1) = referencePoint.x();
 par(2) = referencePoint.y();
 par(3) = referencePoint.z();
 par(4) = mm(1);
 par(5) = mm(2);
 par(6) = mm(3);
 par(7) = mm(4);
 KinematicParameters kPar(par);
 return KinematicState(kPar, kinematicParamError, charge, field);
}

AlgebraicMatrix KinematicPerigeeConversions::jacobianParameters2Kinematic(
	const AlgebraicVector& momentum, const GlobalPoint& referencePoint,
	const TrackCharge& charge, const MagneticField* field)const
{
  PerigeeConversions pc;
  AlgebraicMatrix param2cart = asHepMatrix(pc.jacobianParameters2Cartesian
  	(asSVector<3>(momentum), referencePoint, charge, field));
  AlgebraicMatrix frameTransJ(7, 7, 0);
  for (int i =0;i<6;++i)
    for (int j =0;j<6;++j)
      frameTransJ[i][j] = param2cart[i][j];
  frameTransJ[6][6] = 1;

//   double factor = 1;
//   if (charge != 0){
//    double field = TrackingTools::FakeField::Field::inGeVPerCentimeter(referencePoint).z();
//    factor =  -field*charge;
//    }
//   AlgebraicMatrix frameTransJ(7, 7, 0);
//   frameTransJ[0][0] = 1;
//   frameTransJ[1][1] = 1;
//   frameTransJ[2][2] = 1;
//   frameTransJ[6][6] = 1;
//   frameTransJ[3][3] = - factor * cos(momentum[2])  / (momentum[0]*momentum[0]);
//   frameTransJ[4][3] = - factor * sin(momentum[2])  / (momentum[0]*momentum[0]);
//   frameTransJ[5][3] = - factor / tan(momentum[1])  / (momentum[0]*momentum[0]);
//
//   frameTransJ[3][5] = - factor * sin(momentum[1]) / (momentum[0]);
//   frameTransJ[4][5] =   factor * cos(momentum[1])  / (momentum[0]);
//   frameTransJ[5][4] = -factor/ (momentum[0]*sin(momentum[1])*sin(momentum[1]));

  return frameTransJ;

}


AlgebraicVector
KinematicPerigeeConversions::momentumFromPerigee(const AlgebraicVector& momentum,
	const GlobalPoint& referencePoint, const TrackCharge& ch,
	const MagneticField* field)const
{
 AlgebraicVector mm(4);
 double pt;
 if(ch !=0){
//   pt = abs(MagneticField::inGeVPerCentimeter(referencePoint).z() / momentum[0]);
    pt = std::abs(field->inInverseGeV(referencePoint).z() / momentum[0]);
 }else{pt = 1/ momentum[0];}
 mm(1) = cos(momentum[2]) * pt;
 mm(2) = sin(momentum[2]) * pt;
 mm(3) = pt/tan(momentum[1]);
 mm(4) = momentum(4);
 return mm;
}

