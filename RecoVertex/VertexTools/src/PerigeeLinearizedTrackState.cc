#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h"
#include "RecoVertex/VertexTools/interface/PerigeeRefittedTrackState.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// #include "CommonReco/PatternTools/interface/TransverseImpactPointExtrapolator.h"
// #include "CommonDet/DetUtilities/interface/FastTimeMe.h"

/** Method returning the constant term of the Taylor expansion
 *  of the measurement equation
 */
const AlgebraicVector5 & PerigeeLinearizedTrackState::constantTerm() const
{
  if (!jacobiansAvailable) computeJacobians();
  return theConstantTerm;
}

/**
 * Method returning the Position Jacobian (Matrix A)
 */
const AlgebraicMatrix53 & PerigeeLinearizedTrackState::positionJacobian() const
{
  if (!jacobiansAvailable) computeJacobians();
  return thePositionJacobian;
}

/**      
 * Method returning the Momentum Jacobian (Matrix B)
 */
const AlgebraicMatrix53 & PerigeeLinearizedTrackState::momentumJacobian() const
{
  if (!jacobiansAvailable) computeJacobians();
  return theMomentumJacobian;
}

/** Method returning the parameters of the Taylor expansion
 */
const AlgebraicVector5 & PerigeeLinearizedTrackState::parametersFromExpansion() const
{
  if (!jacobiansAvailable) computeJacobians();
  return theExpandedParams;
}

/**
 * Method returning the TrajectoryStateClosestToPoint at the point
 * of closest approch to the z-axis (a.k.a. transverse impact point)
 */      
const TrajectoryStateClosestToPoint & PerigeeLinearizedTrackState::predictedState() const
{
  if (!jacobiansAvailable) computeJacobians();
  return thePredState;
}

// /** Method returning the impact point measurement     
//  */      
// ImpactPointMeasurement  PerigeeLinearizedTrackState::impactPointMeasurement() const 
// {
//   if (!impactPointAvailable) compute3DImpactPoint(); 
//     return the3DImpactPoint;
// }
// 
void PerigeeLinearizedTrackState::computeJacobians() const
{
  GlobalPoint paramPt(theLinPoint);

//   std::cout << "Track "
//   << "\n Param    " << theTSOS.globalParameters ()
//   << "\n Dir      " << theTSOS.globalDirection ()
//    << "\n";
  thePredState = builder(theTSOS, paramPt); 
  if (!thePredState.isValid())
    return;
//   std::cout << "thePredState " << thePredState.theState().position()<<std::endl;
//   edm::LogInfo("RecoVertex/PerigeeLTS") 
//     << "predstate built" << "\n";
  double field =  theTrack.field()->inInverseGeV(thePredState.theState().position()).z();

  if ((std::abs(theCharge)<1e-5)||(fabs(field)<1.e-10)){
    //neutral track
    computeNeutralJacobians();
  } else {
    //charged track
//     edm::LogInfo("RecoVertex/PerigeeLTS") 
//       << "about to compute charged jac" << "\n";
    computeChargedJacobians();
//     edm::LogInfo("RecoVertex/PerigeeLTS") 
//       << "charged jac computed" << "\n";
  }




  jacobiansAvailable = true;
}

// void PerigeeLinearizedTrackState::compute3DImpactPoint() const 
// {
//   the3DImpactPoint = theIPMExtractor.impactPointMeasurement(theTrack, theLinPoint);
//   impactPointAvailable = true;
// }

bool PerigeeLinearizedTrackState::hasError() const
{
  if (!jacobiansAvailable) computeJacobians();
  return thePredState.hasError();
}

AlgebraicVector5 PerigeeLinearizedTrackState::predictedStateParameters() const
{
  if (!jacobiansAvailable) computeJacobians();
  return thePredState.perigeeParameters().vector();
}
  
AlgebraicVector3 PerigeeLinearizedTrackState::predictedStateMomentumParameters() const
{
  if (!jacobiansAvailable) computeJacobians();
  AlgebraicVector3 momentum;
  momentum[0] = thePredState.perigeeParameters().vector()(0);
  momentum[1] = thePredState.perigeeParameters().theta();
  momentum[2] = thePredState.perigeeParameters().phi();
  return momentum;
}
  
AlgebraicSymMatrix55 PerigeeLinearizedTrackState::predictedStateWeight(int & error) const
{
  if (!jacobiansAvailable) computeJacobians();
  if (!thePredState.isValid()) {
    error = 1;
    return AlgebraicSymMatrix55();
  }
  return thePredState.perigeeError().weightMatrix(error);
}
  
AlgebraicSymMatrix55 PerigeeLinearizedTrackState::predictedStateError() const
{
  if (!jacobiansAvailable) computeJacobians();
  return thePredState.perigeeError().covarianceMatrix();
} 

AlgebraicSymMatrix33 PerigeeLinearizedTrackState::predictedStateMomentumError() const
{
  if (!jacobiansAvailable) computeJacobians();
  return thePredState.perigeeError().covarianceMatrix().Sub<AlgebraicSymMatrix33>(0,2);
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
  AlgebraicVector3 vertexPosition;
  vertexPosition(0) = theRefittedState->position().x();
  vertexPosition(1) = theRefittedState->position().y();
  vertexPosition(2) = theRefittedState->position().z();
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
    
  double thetaAtEP = thePredState.theState().momentum().theta();
  double phiAtEP   = thePredState.theState().momentum().phi();
  double ptAtEP = thePredState.theState().momentum().perp();
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

  AlgebraicVector3 expansionPoint;
  expansionPoint(0) = thePredState.theState().position().x();
  expansionPoint(1) = thePredState.theState().position().y();
  expansionPoint(2) = thePredState.theState().position().z(); 
  AlgebraicVector3 momentumAtExpansionPoint;
  momentumAtExpansionPoint(0) = transverseCurvatureAtEP;  // Transverse Curv
  momentumAtExpansionPoint(1) = thetaAtEP;
  momentumAtExpansionPoint(2) = phiAtEP; 

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

  AlgebraicVector3 expansionPoint;
  expansionPoint(0) = thePredState.theState().position().x();
  expansionPoint(1) = thePredState.theState().position().y();
  expansionPoint(2) = thePredState.theState().position().z(); 
  AlgebraicVector3 momentumAtExpansionPoint;
  momentumAtExpansionPoint(0) = 1 / ptAtEP;  // 
  momentumAtExpansionPoint(1) = thetaAtEP;
  momentumAtExpansionPoint(2) = phiAtEP; 

  theConstantTerm = AlgebraicVector5( theExpandedParams -
  		  thePositionJacobian * expansionPoint -
  		  theMomentumJacobian * momentumAtExpansionPoint );

}

bool PerigeeLinearizedTrackState::isValid() const
{
  if (!theTSOS.isValid())
    return false;

  if (!jacobiansAvailable)
    computeJacobians();

  return jacobiansAvailable;
}
