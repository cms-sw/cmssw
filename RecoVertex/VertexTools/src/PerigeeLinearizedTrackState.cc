#include "RecoVertex/VertexTools/interface/PerigeeLinearizedTrackState.h"
#include "RecoVertex/VertexTools/interface/PerigeeRefittedTrackState.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"
#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "MagneticField/Engine/interface/MagneticField.h"

// #include "CommonReco/PatternTools/interface/TransverseImpactPointExtrapolator.h"
// #include "CommonDet/DetUtilities/interface/FastTimeMe.h"

/** Method returning the constant term of the Taylor expansion
 *  of the measurement equation
 */
AlgebraicVector PerigeeLinearizedTrackState::constantTerm() const
{
  if (!jacobiansAvailable) computeJacobians();
  return theConstantTerm;
}

/**
 * Method returning the Position Jacobian (Matrix A)
 */
AlgebraicMatrix PerigeeLinearizedTrackState::positionJacobian() const
{
  if (!jacobiansAvailable) computeJacobians();
  return thePositionJacobian;
}

/**      
 * Method returning the Momentum Jacobian (Matrix B)
 */
AlgebraicMatrix PerigeeLinearizedTrackState::momentumJacobian() const
{
  if (!jacobiansAvailable) computeJacobians();
  return theMomentumJacobian;
}

/** Method returning the parameters of the Taylor expansion
 */
AlgebraicVector PerigeeLinearizedTrackState::parametersFromExpansion() const
{
  if (!jacobiansAvailable) computeJacobians();
  return theExpandedParams;
}

/**
 * Method returning the TrajectoryStateClosestToPoint at the point
 * of closest approch to the z-axis (a.k.a. transverse impact point)
 */      
TrajectoryStateClosestToPoint PerigeeLinearizedTrackState::predictedState() const
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
//change to allow multiple state:  thePredState = builder(theTrack.innermostState(), paramPt); 

//   TransverseImpactPointExtrapolator extr;
//   FastTimer buildTimer, extrTimer, allTimer;
// for (int i=0;i<50000;++i){
//   {
//     FastTimeMe<FastTimer> bTimer(allTimer);
//   TrajectoryStateOnSurface tsos = extr.extrapolate(theTSOS, paramPt);
//   thePredState = builder(tsos, paramPt); 
// //  cout << "Track position from Extrapolator: "<<tsos.globalPosition()<<endl;
//   }
//   {
//     FastTimeMe<FastTimer> aTimer(buildTimer);
// //  cout << "Track position from TSCP builder: "<<thePredState.theState().position()<<endl;
//   }
// 
//   {
//     FastTimeMe<FastTimer> bTimer(extrTimer);
//   TrajectoryStateOnSurface tsos = extr.extrapolate(theTSOS, paramPt);
// //  cout << "Track position from Extrapolator: "<<tsos.globalPosition()<<endl;
//   }
// 
// }
//   cout << endl << buildTimer.average().ticks() 
//        << " TSCP builder time per event (in clock ticks)" << endl;
//   cout << endl << extrTimer.average().ticks() 
//        << " Extrapolator time per event (in clock ticks)" << endl;
//   cout << endl << allTimer.average().ticks() 
//        << " All          time per event (in clock ticks)" << endl;

  thePredState = builder(theTSOS, paramPt); 
  if (std::abs(theCharge)<1e-5) {
    //neutral track
    computeNeutralJacobians();
  } else {
    //charged track
    computeChargedJacobians();
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

AlgebraicVector PerigeeLinearizedTrackState::predictedStateParameters() const
{
  if (!jacobiansAvailable) computeJacobians();
  return thePredState.perigeeParameters().vector();
}
  
AlgebraicVector PerigeeLinearizedTrackState::predictedStateMomentumParameters() const
{
  if (!jacobiansAvailable) computeJacobians();
  AlgebraicVector momentum(3);
  momentum[0] = thePredState.perigeeParameters().vector()[0];
  momentum[1] = thePredState.perigeeParameters().theta();
  momentum[2] = thePredState.perigeeParameters().phi();
  return momentum;
}
  
AlgebraicSymMatrix PerigeeLinearizedTrackState::predictedStateWeight() const
{
  if (!jacobiansAvailable) computeJacobians();
  return thePredState.perigeeError().weightMatrix();
}
  
AlgebraicSymMatrix PerigeeLinearizedTrackState::predictedStateError() const
{
  if (!jacobiansAvailable) computeJacobians();
  return thePredState.perigeeError().covarianceMatrix();
} 

AlgebraicSymMatrix PerigeeLinearizedTrackState::predictedStateMomentumError() const
{
  if (!jacobiansAvailable) computeJacobians();
  return thePredState.perigeeError().covarianceMatrix().sub(1,3);
} 

bool PerigeeLinearizedTrackState::operator ==(LinearizedTrackState& other)const
{
  const PerigeeLinearizedTrackState* otherP = 
  	dynamic_cast<const PerigeeLinearizedTrackState*>(&other);
  if (otherP == 0) {
   throw VertexException("PerigeeLinearizedTrackState: don't know how to compare myself to non-perigee track state");
  }
  return (otherP->track() == theTrack);
}


bool PerigeeLinearizedTrackState::operator ==(ReferenceCountingPointer<LinearizedTrackState>& other)const
{
  const PerigeeLinearizedTrackState* otherP = 
  	dynamic_cast<const PerigeeLinearizedTrackState*>(other.get());
  if (otherP == 0) {
   throw VertexException("PerigeeLinearizedTrackState: don't know how to compare myself to non-perigee track state");
  }
  return (otherP->track() == theTrack);
}


RefCountedLinearizedTrackState
PerigeeLinearizedTrackState::stateWithNewLinearizationPoint
			(const GlobalPoint & newLP) const
{
  return RefCountedLinearizedTrackState(
  		new PerigeeLinearizedTrackState(newLP, track(), theTSOS));
}

RefCountedRefittedTrackState
PerigeeLinearizedTrackState::createRefittedTrackState(
  	const GlobalPoint & vertexPosition, 
	const AlgebraicVector & vectorParameters,
	const AlgebraicSymMatrix & covarianceMatrix) const
{
  PerigeeConversions perigeeConversions;
  TrajectoryStateClosestToPoint refittedTSCP = 
        perigeeConversions.trajectoryStateClosestToPoint(
	  vectorParameters, vertexPosition, charge(), covarianceMatrix);
  return RefCountedRefittedTrackState(new PerigeeRefittedTrackState(refittedTSCP));
}

std::vector< RefCountedLinearizedTrackState > 
PerigeeLinearizedTrackState::components() const
{
  std::vector<RefCountedLinearizedTrackState> result; result.reserve(1);
  result.push_back(RefCountedLinearizedTrackState( 
  			const_cast<PerigeeLinearizedTrackState*>(this)));
  return result;
}


void PerigeeLinearizedTrackState::computeChargedJacobians() const
{
  GlobalPoint paramPt(theLinPoint);
  //tarjectory parameters
  double field =  theTSOS.freeState()->parameters().magneticField().inTesla(thePredState.theState().position()).z() * 2.99792458e-3;
//   MagneticField::inInverseGeV(thePredState.theState().position()).z();
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

  AlgebraicVector trackParameterFromExpansionPoint(5);
  trackParameterFromExpansionPoint[0] = transverseCurvatureAtEP;
  trackParameterFromExpansionPoint[1] = thetaAtEP;
  trackParameterFromExpansionPoint[3] = 1/transverseCurvatureAtEP  - signTC * S;
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
  trackParameterFromExpansionPoint[2] = phiFEP;
  trackParameterFromExpansionPoint[4] = z_v - paramPt.z() - 
  	(phiAtEP - trackParameterFromExpansionPoint[2]) / tan(thetaAtEP)/transverseCurvatureAtEP;
		
  // The Jacobian: (all at the expansion point)
  // [i,j]
  // i = 0: rho , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
  // j = 0: x_v, 1: y_v, 2: z_v
  thePositionJacobian = AlgebraicMatrix(5,3,0);

  thePositionJacobian[2][0] = - Y / (SS);
  thePositionJacobian[2][1] = X / (SS);
  thePositionJacobian[3][0] = - signTC * X / S;
  thePositionJacobian[3][1] = - signTC * Y / S;
  thePositionJacobian[4][0] = thePositionJacobian[2][0]/tan(thetaAtEP)/transverseCurvatureAtEP;
  thePositionJacobian[4][1] = thePositionJacobian[2][1]/tan(thetaAtEP)/transverseCurvatureAtEP;
  thePositionJacobian[4][2] = 1;

  // [i,j]
  // i = 0: rho , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
  // j = 0: rho, 1: theta, 2: phi_v
  theMomentumJacobian = AlgebraicMatrix(5,3,0);
  theMomentumJacobian[0][0] = 1;
  theMomentumJacobian[1][1] = 1;

  theMomentumJacobian[2][0] = -
  	(X*cos(phiAtEP) + Y*sin(phiAtEP))/
	(SS*transverseCurvatureAtEP*transverseCurvatureAtEP);

  theMomentumJacobian[2][2] = (Y*cos(phiAtEP) - X*sin(phiAtEP)) / 
  	(SS*transverseCurvatureAtEP);

  theMomentumJacobian[3][0] = 
  	(signTC * (Y*cos(phiAtEP) - X*sin(phiAtEP)) / S - 1)/
	(transverseCurvatureAtEP*transverseCurvatureAtEP);
  
  theMomentumJacobian[3][2] = signTC *(X*cos(phiAtEP) + Y*sin(phiAtEP))/
  	(S*transverseCurvatureAtEP);
  
  theMomentumJacobian[4][0] = (phiAtEP - trackParameterFromExpansionPoint[2]) /
  	tan(thetaAtEP)/(transverseCurvatureAtEP*transverseCurvatureAtEP)+
	theMomentumJacobian[2][0] / tan(thetaAtEP)/transverseCurvatureAtEP;

  theMomentumJacobian[4][1] = (phiAtEP - trackParameterFromExpansionPoint[2]) *
  	(1 + 1/(tan(thetaAtEP)*tan(thetaAtEP)))/transverseCurvatureAtEP;

  theMomentumJacobian[4][2] = (theMomentumJacobian[2][2] - 1) / 
  				tan(thetaAtEP)/transverseCurvatureAtEP;

   // And finally the residuals:

  AlgebraicVector expansionPoint(3);
  expansionPoint[0] = thePredState.theState().position().x();
  expansionPoint[1] = thePredState.theState().position().y();
  expansionPoint[2] = thePredState.theState().position().z(); 
  AlgebraicVector momentumAtExpansionPoint(3);
  momentumAtExpansionPoint[0] = transverseCurvatureAtEP;  // Transverse Curv
  momentumAtExpansionPoint[1] = thetaAtEP;
  momentumAtExpansionPoint[2] = phiAtEP; 

  theExpandedParams = trackParameterFromExpansionPoint;


  theConstantTerm = AlgebraicVector( trackParameterFromExpansionPoint -
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

  AlgebraicVector trackParameterFromExpansionPoint(5);
  trackParameterFromExpansionPoint[0] = 1 / ptAtEP;
  trackParameterFromExpansionPoint[1] = thetaAtEP;
  trackParameterFromExpansionPoint[2] = phiAtEP;
  trackParameterFromExpansionPoint[3] = X*sin(phiAtEP) - Y*cos(phiAtEP);
  trackParameterFromExpansionPoint[4] = z_v - paramPt.z() - 
  	(X*cos(phiAtEP) + Y*sin(phiAtEP)) / tan(thetaAtEP);

  // The Jacobian: (all at the expansion point)
  // [i,j]
  // i = 0: rho = 1/pt , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
  // j = 0: x_v, 1: y_v, 2: z_v
  thePositionJacobian = AlgebraicMatrix(5,3,0);

  thePositionJacobian[3][0] =   sin(phiAtEP);
  thePositionJacobian[3][1] = - cos(phiAtEP);
  thePositionJacobian[4][0] = - cos(phiAtEP)/tan(thetaAtEP);
  thePositionJacobian[4][1] = - sin(phiAtEP)/tan(thetaAtEP);
  thePositionJacobian[4][2] = 1;

  // [i,j]
  // i = 0: rho = 1/pt , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
  // j = 0: rho = 1/pt , 1: theta, 2: phi_v
  theMomentumJacobian = AlgebraicMatrix(5,3,0);
  theMomentumJacobian[0][0] = 1;
  theMomentumJacobian[1][1] = 1;
  theMomentumJacobian[2][2] = 1;

  theMomentumJacobian[3][2] = X*cos(phiAtEP) + Y*sin(phiAtEP);
  
  theMomentumJacobian[4][1] = theMomentumJacobian[3][2]*
  	(1 + 1/(tan(thetaAtEP)*tan(thetaAtEP)));

  theMomentumJacobian[4][2] = (X*sin(phiAtEP) - Y*cos(phiAtEP))/tan(thetaAtEP);

   // And finally the residuals:

  AlgebraicVector expansionPoint(3);
  expansionPoint[0] = thePredState.theState().position().x();
  expansionPoint[1] = thePredState.theState().position().y();
  expansionPoint[2] = thePredState.theState().position().z(); 
  AlgebraicVector momentumAtExpansionPoint(3);
  momentumAtExpansionPoint[0] = 1 / ptAtEP;  // 
  momentumAtExpansionPoint[1] = thetaAtEP;
  momentumAtExpansionPoint[2] = phiAtEP; 

  theExpandedParams = trackParameterFromExpansionPoint;


  theConstantTerm = AlgebraicVector( trackParameterFromExpansionPoint -
  		  thePositionJacobian * expansionPoint -
  		  theMomentumJacobian * momentumAtExpansionPoint );


}
