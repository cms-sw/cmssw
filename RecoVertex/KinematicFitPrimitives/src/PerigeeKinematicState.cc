#include "RecoVertex/KinematicFitPrimitives/interface/PerigeeKinematicState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicPerigeeConversions.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

PerigeeKinematicState::PerigeeKinematicState(const KinematicState& state, const GlobalPoint& pt):
 point(pt), inState(state),  errorIsAvailable(true),vl(true)
{
 if(!(state.isValid())) throw VertexException("PerigeeKinematicState::kinematic state passed is not valid!");

//working with parameters:
 KinematicPerigeeConversions conversions;
 par  = conversions.extendedPerigeeFromKinematicParameters(state,pt);
   
//creating the error
 AlgebraicSymMatrix err = state.kinematicParametersError().matrix();

//making jacobian for curvilinear frame
 JacobianCartesianToCurvilinear jj(state.freeTrajectoryState().parameters());  
 AlgebraicMatrix ki2cu(6,7,0);
 ki2cu.sub(1,1,jj.jacobian());
 ki2cu(6,7) = 1.;
 AlgebraicMatrix cu2pe(6,6,0);
 PerigeeConversions pc;
 cu2pe.sub(1,1,pc.jacobianCurvilinear2Perigee(state.freeTrajectoryState()));
 cu2pe(6,6) = 1.;
 cu2pe = cu2pe*ki2cu;
 err = err.similarity(cu2pe);
 cov = ExtendedPerigeeTrajectoryError(err);
}

/*
AlgebraicMatrix PerigeeKinematicState::jacobianKinematicToExPerigee(const KinematicState& state,
                                                                    const GlobalPoint& pt)const
{
 
 AlgebraicMatrix jac(6,7,0);
 jac(6,7) = 1;
 jac(5,3) = 1;
 AlgebraicVector par = state.kinematicParameters().vector();
 GlobalVector impactDistance = state.globalPosition() - point;
 double field = MagneticField::inInverseGeV(state.globalPosition()).z();
 double signTC = -state.particleCharge();
 double theta = state.globalMomentum().theta();
 double phi = state.globalMomentum().phi();
 double ptr  = state.globalMomentum().transverse();
 double transverseCurvature = field/ptr*signTC;
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

//jacobian corrections

//  jac(1,4) = -(field*signTC/(transverseCurvature*transverseCurvature))* cos(phi);
//  jac(1,5) = -(field*signTC/(transverseCurvature*transverseCurvature))* sin(phi);
//  jac(1,6) = -(field*signTC/(transverseCurvature*transverseCurvature))*tan(theta);
 
 jac(1,4) = (1/ptr*signTC) * cos(phi);
 jac(1,5) = (1/ptr*signTC) * sin(phi);
 jac(1,6) = (1/ptr*signTC) * tan(theta);
 
 jac(2,6) = (ptr)/(cos(theta) * cos(theta));
 jac(3,1) = - epsilon * cos(phi);
 jac(3,2) = - epsilon * sin(phi);


// jac(3,4) = 
 jac(3,4) = - ptr * sin(phi);
 jac(3,5) =  ptr * cos(phi);
 jac(4,1) = - sin(phi);
 jac(4,2) =  cos(phi);
 return jac;
 
}



AlgebraicMatrix PerigeeKinematicState::jacobianExPerigeeToKinematic(const ExtendedPerigeeTrajectoryParameters& state, 
                                                                    const GlobalPoint& point)const
{

 AlgebraicMatrix jac(7,6,0);
 return jac;
 
}
*/
//temporary method move
