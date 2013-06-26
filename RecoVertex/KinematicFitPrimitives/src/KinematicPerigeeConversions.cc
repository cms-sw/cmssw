#include "RecoVertex/KinematicFitPrimitives/interface/KinematicPerigeeConversions.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"


ExtendedPerigeeTrajectoryParameters
KinematicPerigeeConversions::extendedPerigeeFromKinematicParameters(
	const KinematicState& state, const GlobalPoint& point) const
{
//making an extended perigee parametrization
//out of kinematic state and point defined
 AlgebraicVector6 res;
 res(5) = state.mass();
//  AlgebraicVector7 par = state.kinematicParameters().vector();
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
   res(0) = field / pt*signTC;
 } else {
   res(0) = 1 / pt;
 }

 res(1) = theta;
 res(2) = phi;
 res(3) = epsilon;
 res(4) = impactDistance.z();
 return ExtendedPerigeeTrajectoryParameters(res,state.particleCharge());
}

KinematicParameters KinematicPerigeeConversions::kinematicParametersFromExPerigee(
	const ExtendedPerigeeTrajectoryParameters& pr, const GlobalPoint& point,
	const MagneticField* field) const
{
 AlgebraicVector7 par;
 AlgebraicVector6 theVector = pr.vector();
 double pt;
 if(pr.charge() !=0){
  pt = std::abs(field->inInverseGeV(point).z() / theVector(1));
 }else{pt = 1/theVector(1);}
 par(6) = theVector[5];
 par(0) = theVector[3]*sin(theVector[2])+point.x();
 par(1) = -theVector[3]*cos(theVector[2])+point.y();
 par(2) = theVector[4]+point.z();
 par(3) = cos(theVector[2]) * pt;
 par(4) = sin(theVector[2]) * pt;
 par(5) = pt/tan(theVector[1]);

 return KinematicParameters(par);
}


KinematicState
KinematicPerigeeConversions::kinematicState(const AlgebraicVector4& momentum,
	const GlobalPoint& referencePoint, const TrackCharge& charge,
	const AlgebraicSymMatrix77& theCovarianceMatrix, const MagneticField* field) const
{
 AlgebraicMatrix77 param2cart = jacobianParameters2Kinematic(momentum,
				referencePoint, charge, field);
 AlgebraicSymMatrix77 kinematicErrorMatrix = ROOT::Math::Similarity(param2cart,theCovarianceMatrix);
//  kinematicErrorMatrix.assign(param2cart*theCovarianceMatrix*param2cart.T());

 KinematicParametersError kinematicParamError(kinematicErrorMatrix);
 AlgebraicVector7 par;
 AlgebraicVector4 mm = momentumFromPerigee(momentum, referencePoint, charge, field);
 par(0) = referencePoint.x();
 par(1) = referencePoint.y();
 par(2) = referencePoint.z();
 par(3) = mm(0);
 par(4) = mm(1);
 par(5) = mm(2);
 par(6) = mm(3);
 KinematicParameters kPar(par);
 return KinematicState(kPar, kinematicParamError, charge, field);
}

AlgebraicMatrix77 KinematicPerigeeConversions::jacobianParameters2Kinematic(
	const AlgebraicVector4& momentum, const GlobalPoint& referencePoint,
	const TrackCharge& charge, const MagneticField* field)const
{
  AlgebraicMatrix66 param2cart = PerigeeConversions::jacobianParameters2Cartesian
  	(AlgebraicVector3(momentum[0],momentum[1],momentum[2]),
	referencePoint, charge, field);
  AlgebraicMatrix77 frameTransJ;
  for (int i =0;i<6;++i)
    for (int j =0;j<6;++j)
      frameTransJ(i, j) = param2cart(i, j);
  frameTransJ(6, 6) = 1;

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

// Cartesian (px,py,px,m) from extended perigee
AlgebraicVector4
KinematicPerigeeConversions::momentumFromPerigee(const AlgebraicVector4& momentum,
	const GlobalPoint& referencePoint, const TrackCharge& ch,
	const MagneticField* field)const
{
 AlgebraicVector4 mm;
 double pt;
 if(ch !=0){
//   pt = abs(MagneticField::inGeVPerCentimeter(referencePoint).z() / momentum[0]);
    pt = std::abs(field->inInverseGeV(referencePoint).z() / momentum[0]);
 }else{pt = 1/ momentum[0];}
 mm(0) = cos(momentum[2]) * pt;
 mm(1) = sin(momentum[2]) * pt;
 mm(2) = pt/tan(momentum[1]);
 mm(3) = momentum[3];
 return mm;
}

