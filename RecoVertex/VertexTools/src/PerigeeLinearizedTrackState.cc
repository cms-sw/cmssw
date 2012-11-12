#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h"
#include "RecoVertex/VertexTools/interface/PerigeeRefittedTrackState.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



void PerigeeLinearizedTrackState::computeJacobians() const
{
  GlobalPoint paramPt(theLinPoint);
  
  thePredState = builder(theTSOS, paramPt); 
  if unlikely(!thePredState.isValid()) return;
  
  double field =  theTrack.field()->inInverseGeV(thePredState.theState().position()).z();
  
  if ((std::abs(theCharge)<1e-5)||(fabs(field)<1.e-10)){
    //neutral track
    computeNeutralJacobians();
  } else {
    //charged track
    computeChargedJacobians();
  }

  jacobiansAvailable = true;
}



bool PerigeeLinearizedTrackState::operator ==(LinearizedTrackState<5> & other)const
{
  const PerigeeLinearizedTrackState* otherP = 
  	dynamic_cast<const PerigeeLinearizedTrackState*>(&other);
  if (otherP == 0) {
   throw VertexException("PerigeeLinearizedTrackState: don't know how to compare myself to non-perigee track state");
  }
  return (otherP->track() == theTrack);
}


bool PerigeeLinearizedTrackState::operator ==(ReferenceCountingPointer<LinearizedTrackState<5> >& other)const
{
  const PerigeeLinearizedTrackState* otherP = 
  	dynamic_cast<const PerigeeLinearizedTrackState*>(other.get());
  if (otherP == 0) {
   throw VertexException("PerigeeLinearizedTrackState: don't know how to compare myself to non-perigee track state");
  }
  return (otherP->track() == theTrack);
}


PerigeeLinearizedTrackState::RefCountedLinearizedTrackState
PerigeeLinearizedTrackState::stateWithNewLinearizationPoint
      (const GlobalPoint & newLP) const
{
 return RefCountedLinearizedTrackState(
 		new PerigeeLinearizedTrackState(newLP, track(), theTSOS));
}

PerigeeLinearizedTrackState::RefCountedRefittedTrackState
PerigeeLinearizedTrackState::createRefittedTrackState(
  	const GlobalPoint & vertexPosition, 
	const AlgebraicVector3 & vectorParameters,
	const AlgebraicSymMatrix66 & covarianceMatrix) const
{
  TrajectoryStateClosestToPoint refittedTSCP = 
        PerigeeConversions::trajectoryStateClosestToPoint(
	  vectorParameters, vertexPosition, charge(), covarianceMatrix, theTrack.field());
  return RefCountedRefittedTrackState(new PerigeeRefittedTrackState(refittedTSCP, vectorParameters));
}

std::vector< PerigeeLinearizedTrackState::RefCountedLinearizedTrackState > 
PerigeeLinearizedTrackState::components() const
{
  std::vector<RefCountedLinearizedTrackState> result; result.reserve(1);
  result.push_back(RefCountedLinearizedTrackState( 
  			const_cast<PerigeeLinearizedTrackState*>(this)));
  return result;
}


AlgebraicVector5 PerigeeLinearizedTrackState::refittedParamFromEquation(
	const RefCountedRefittedTrackState & theRefittedState) const
{
  auto p = theRefittedState->position();
  AlgebraicVector3 vertexPosition(p.x(),p.y(),p.z());
  AlgebraicVector3 momentum = theRefittedState->momentumVector();
  if ((momentum(2)*predictedStateMomentumParameters()(2) <  0)&&(fabs(momentum(2))>M_PI/2) ) {
    if (predictedStateMomentumParameters()(2) < 0.) momentum(2)-= 2*M_PI;
    if (predictedStateMomentumParameters()(2) > 0.) momentum(2)+= 2*M_PI;
  }
  AlgebraicVectorN param = constantTerm() + 
		       positionJacobian() * vertexPosition +
		       momentumJacobian() * momentum;
  if (param(2) >  M_PI) param(2)-= 2*M_PI;
  if (param(2) < -M_PI) param(2)+= 2*M_PI;

  return param;
}


void PerigeeLinearizedTrackState::checkParameters(AlgebraicVector5 & parameters) const
{
  if (parameters(2) >  M_PI) parameters(2)-= 2*M_PI;
  if (parameters(2) < -M_PI) parameters(2)+= 2*M_PI;
}

void PerigeeLinearizedTrackState::computeChargedJacobians() const
{
  GlobalPoint paramPt(theLinPoint);
  //tarjectory parameters
  double field =  theTrack.field()->inInverseGeV(thePredState.theState().position()).z();
  double signTC = -theCharge;
    
  double thetaAtEP = thePredState.perigeeParameters().theta();
  double phiAtEP   = thePredState.perigeeParameters().phi();
  double ptAtEP = thePredState.pt();
  double transverseCurvatureAtEP = field / ptAtEP*signTC;

  double x_v = thePredState.theState().position().x();
  double y_v = thePredState.theState().position().y();
  double z_v = thePredState.theState().position().z();
  double X = x_v - paramPt.x() - sin(phiAtEP) / transverseCurvatureAtEP;
  double Y = y_v - paramPt.y() + cos(phiAtEP) / transverseCurvatureAtEP;
  double SS = X*X + Y*Y;
  double S = sqrt(SS);

  // The track parameters at the expansion point

  theExpandedParams[0] = transverseCurvatureAtEP;
  theExpandedParams[1] = thetaAtEP;
  theExpandedParams[3] = 1/transverseCurvatureAtEP  - signTC * S;
  double phiFEP;
  if (std::abs(X)>std::abs(Y)) {
    double signX = (X>0.0? +1.0:-1.0);
    phiFEP = -signTC * signX*acos(signTC*Y/S);
  } else {
    phiFEP = asin(-signTC*X/S);
    if ((signTC*Y)<0.0)
      phiFEP = M_PI - phiFEP;
  }
  if (phiFEP>M_PI) phiFEP-= 2*M_PI;
  theExpandedParams[2] = phiFEP;
  theExpandedParams[4] = z_v - paramPt.z() - 
  	(phiAtEP - theExpandedParams[2]) / tan(thetaAtEP)/transverseCurvatureAtEP;
		
  // The Jacobian: (all at the expansion point)
  // [i,j]
  // i = 0: rho , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
  // j = 0: x_v, 1: y_v, 2: z_v

  thePositionJacobian(2,0) = - Y / (SS);
  thePositionJacobian(2,1) = X / (SS);
  thePositionJacobian(3,0) = - signTC * X / S;
  thePositionJacobian(3,1) = - signTC * Y / S;
  thePositionJacobian(4,0) = thePositionJacobian(2,0)/tan(thetaAtEP)/transverseCurvatureAtEP;
  thePositionJacobian(4,1) = thePositionJacobian(2,1)/tan(thetaAtEP)/transverseCurvatureAtEP;
  thePositionJacobian(4,2) = 1;

  // [i,j]
  // i = 0: rho , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
  // j = 0: rho, 1: theta, 2: phi_v
  theMomentumJacobian(0,0) = 1;
  theMomentumJacobian(1,1) = 1;

  theMomentumJacobian(2,0) = -
  	(X*cos(phiAtEP) + Y*sin(phiAtEP))/
	(SS*transverseCurvatureAtEP*transverseCurvatureAtEP);

  theMomentumJacobian(2,2) = (Y*cos(phiAtEP) - X*sin(phiAtEP)) / 
  	(SS*transverseCurvatureAtEP);

  theMomentumJacobian(3,0) = 
  	(signTC * (Y*cos(phiAtEP) - X*sin(phiAtEP)) / S - 1)/
	(transverseCurvatureAtEP*transverseCurvatureAtEP);
  
  theMomentumJacobian(3,2) = signTC *(X*cos(phiAtEP) + Y*sin(phiAtEP))/
  	(S*transverseCurvatureAtEP);
  
  theMomentumJacobian(4,0) = (phiAtEP - theExpandedParams[2]) /
  	tan(thetaAtEP)/(transverseCurvatureAtEP*transverseCurvatureAtEP)+
	theMomentumJacobian(2,0) / tan(thetaAtEP)/transverseCurvatureAtEP;

  theMomentumJacobian(4,1) = (phiAtEP - theExpandedParams[2]) *
  	(1 + 1/(tan(thetaAtEP)*tan(thetaAtEP)))/transverseCurvatureAtEP;

  theMomentumJacobian(4,2) = (theMomentumJacobian(2,2) - 1) / 
  				tan(thetaAtEP)/transverseCurvatureAtEP;

   // And finally the residuals:

  auto p = thePredState.theState().position();
  AlgebraicVector3 expansionPoint(p.x(),p.y(),p.z());
  AlgebraicVector3 momentumAtExpansionPoint( transverseCurvatureAtEP,thetaAtEP,phiAtEP); 

  theConstantTerm = AlgebraicVector5( theExpandedParams -
  		  thePositionJacobian * expansionPoint -
  		  theMomentumJacobian * momentumAtExpansionPoint );

}





void PerigeeLinearizedTrackState::computeNeutralJacobians() const
{
  GlobalPoint paramPt(theLinPoint);

  //tarjectory parameters
  double thetaAtEP = thePredState.theState().momentum().theta();
  double phiAtEP   = thePredState.theState().momentum().phi();
  double ptAtEP = thePredState.theState().momentum().perp();

  double x_v = thePredState.theState().position().x();
  double y_v = thePredState.theState().position().y();
  double z_v = thePredState.theState().position().z();
  double X = x_v - paramPt.x();
  double Y = y_v - paramPt.y();

  // The track parameters at the expansion point

  theExpandedParams(0) = 1 / ptAtEP;
  theExpandedParams(1) = thetaAtEP;
  theExpandedParams(2) = phiAtEP;
  theExpandedParams(3) = X*sin(phiAtEP) - Y*cos(phiAtEP);
  theExpandedParams(4) = z_v - paramPt.z() - 
  	(X*cos(phiAtEP) + Y*sin(phiAtEP)) / tan(thetaAtEP);

  // The Jacobian: (all at the expansion point)
  // [i,j]
  // i = 0: rho = 1/pt , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
  // j = 0: x_v, 1: y_v, 2: z_v

  thePositionJacobian(3,0) =   sin(phiAtEP);
  thePositionJacobian(3,1) = - cos(phiAtEP);
  thePositionJacobian(4,0) = - cos(phiAtEP)/tan(thetaAtEP);
  thePositionJacobian(4,1) = - sin(phiAtEP)/tan(thetaAtEP);
  thePositionJacobian(4,2) = 1;

  // [i,j]
  // i = 0: rho = 1/pt , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
  // j = 0: rho = 1/pt , 1: theta, 2: phi_v

  theMomentumJacobian(0,0) = 1;
  theMomentumJacobian(1,1) = 1;
  theMomentumJacobian(2,2) = 1;

  theMomentumJacobian(3,2) = X*cos(phiAtEP) + Y*sin(phiAtEP);
  
  theMomentumJacobian(4,1) = theMomentumJacobian(3,2)*
  	(1 + 1/(tan(thetaAtEP)*tan(thetaAtEP)));

  theMomentumJacobian(4,2) = (X*sin(phiAtEP) - Y*cos(phiAtEP))/tan(thetaAtEP);

   // And finally the residuals:

  auto p = thePredState.theState().position();
  AlgebraicVector3 expansionPoint(p.x(),p.y(),p.z());
  AlgebraicVector3 momentumAtExpansionPoint(1./ptAtEP,thetaAtEP,phiAtEP); 

  theConstantTerm = AlgebraicVector5( theExpandedParams -
  		  thePositionJacobian * expansionPoint -
  		  theMomentumJacobian * momentumAtExpansionPoint );

}

