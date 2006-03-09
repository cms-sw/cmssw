#include "RecoVertex/KinematicFitPrimitives/interface/ParticleKinematicLinearizedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicRefittedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicPerigeeConversions.h"
#include "TrackingTools/TrajectoryState/interface/FakeField.h"

AlgebraicVector ParticleKinematicLinearizedTrackState::constantTerm() const
{
 if (!jacobiansAvailable) computeJacobians();
 return theConstantTerm;
}

AlgebraicMatrix ParticleKinematicLinearizedTrackState::positionJacobian() const
{
 if (!jacobiansAvailable) computeJacobians();
 return thePositionJacobian;
}

AlgebraicMatrix ParticleKinematicLinearizedTrackState::momentumJacobian() const
{
 if (!jacobiansAvailable)computeJacobians();
 return theMomentumJacobian;
}

AlgebraicVector ParticleKinematicLinearizedTrackState::parametersFromExpansion() const
{
 if (!jacobiansAvailable) computeJacobians();
 return theExpandedParams;
}

AlgebraicVector ParticleKinematicLinearizedTrackState::predictedStateParameters() const
{
 if(!jacobiansAvailable) computeJacobians();
// cout<<"Kinematic predicted state parameters: "<<thePredState.perigeeParameters().vector()<<endl;
 return thePredState.perigeeParameters().vector();
}
  
AlgebraicSymMatrix  ParticleKinematicLinearizedTrackState::predictedStateWeight() const
{
 if(!jacobiansAvailable) computeJacobians();
 return thePredState.perigeeError().weightMatrix();
}
  
AlgebraicSymMatrix  ParticleKinematicLinearizedTrackState::predictedStateError() const
{
 if(!jacobiansAvailable) computeJacobians();
 return thePredState.perigeeError().covarianceMatrix();
} 
   
// ImpactPointMeasurement  ParticleKinematicLinearizedTrackState::impactPointMeasurement() const
// { throw VertexException(" ParticleKinematicLinearizedTrackState::impact point measurement is not implemented for kinematic classes!");}
  
TrackCharge  ParticleKinematicLinearizedTrackState::charge() const
{return part->initialState().particleCharge();}

RefCountedKinematicParticle  ParticleKinematicLinearizedTrackState::particle() const
{return part;}

bool  ParticleKinematicLinearizedTrackState::operator ==(LinearizedTrackState& other)const
{
 const  ParticleKinematicLinearizedTrackState* otherP = 
  	dynamic_cast<const  ParticleKinematicLinearizedTrackState*>(&other);
   if (otherP == 0) {
   throw VertexException(" ParticleKinematicLinearizedTrackState:: don't know how to compare myself to non-kinematic track state");
  }
  return (*(otherP->particle()) == *part);}

bool  ParticleKinematicLinearizedTrackState::hasError() const
{
 if (!jacobiansAvailable) computeJacobians();
 return thePredState.isValid();
}


// here we make a and b matrices of
// our measurement function expansion.
// marices will be almost the same as for 
// classical perigee, but bigger: 
// (6x3) and (6x4) respectivelly.
void  ParticleKinematicLinearizedTrackState::computeJacobians() const
{
 GlobalPoint paramPt(theLinPoint);
 thePredState = builder(part->currentState(), paramPt); 
 
 
 if (abs(theCharge)<1e-5) {

//neutral track
  computeNeutralJacobians();
 }else{

//charged track
  computeChargedJacobians();
 } 
 jacobiansAvailable = true;
}
ReferenceCountingPointer<LinearizedTrackState>  ParticleKinematicLinearizedTrackState::stateWithNewLinearizationPoint
  	                                           (const GlobalPoint & newLP) const
{
 RefCountedKinematicParticle pr = part;
 return new  ParticleKinematicLinearizedTrackState(newLP, pr);
}
						   
RefCountedRefittedTrackState  ParticleKinematicLinearizedTrackState::createRefittedTrackState(
                                                   const GlobalPoint & vertexPosition, 
	                                           const AlgebraicVector & vectorParameters,
	                                           const AlgebraicSymMatrix & covarianceMatrix)const
{
 KinematicPerigeeConversions conversions;  
 KinematicState lst = conversions.kinematicState(vectorParameters,vertexPosition,
                                                     charge(),covarianceMatrix); 
 RefCountedRefittedTrackState rst =  RefCountedRefittedTrackState(new KinematicRefittedTrackState(lst));  
 return rst;
}						   
	
AlgebraicVector  ParticleKinematicLinearizedTrackState::predictedStateMomentumParameters() const
{
 if(!jacobiansAvailable) computeJacobians();
 AlgebraicVector res(4);
 res(1) = thePredState.perigeeParameters().vector()(1);
 res(2) = thePredState.perigeeParameters().vector()(2);
 res(3) = thePredState.perigeeParameters().vector()(3);
 res(4) = thePredState.perigeeParameters().vector()(6);
 return res;
} 
 
AlgebraicSymMatrix  ParticleKinematicLinearizedTrackState::predictedStateMomentumError() const
{ 
 if(!jacobiansAvailable) computeJacobians();
 AlgebraicSymMatrix res(4,0);
 AlgebraicSymMatrix m3 = thePredState.perigeeError().weightMatrix().sub(1,3);
 res.sub(1,m3);
 res(4,4) = thePredState.perigeeError().weightMatrix()(6,6);
 res(4,1) = thePredState.perigeeError().weightMatrix()(6,1);
 res(4,2) = thePredState.perigeeError().weightMatrix()(6,2);
 res(4,3) = thePredState.perigeeError().weightMatrix()(6,3);
 return res;
}						   
						   
double  ParticleKinematicLinearizedTrackState::weightInMixture() const
{return 1.;}


std::vector<ReferenceCountingPointer<LinearizedTrackState> >  ParticleKinematicLinearizedTrackState::components()const
{
 std::vector<ReferenceCountingPointer<LinearizedTrackState> > res;
 res.reserve(1);
 res.push_back(RefCountedLinearizedTrackState( 
  			const_cast< ParticleKinematicLinearizedTrackState*>(this)));
 return res;
}

void ParticleKinematicLinearizedTrackState::computeChargedJacobians() const
{
 GlobalPoint paramPt(theLinPoint);
// thePredState = builder(part->currentState(), paramPt);
 
 double field = TrackingTools::FakeField::Field::inInverseGeV(thePredState.theState().globalPosition()).z();
 double signTC = -part->currentState().particleCharge();
 
 double thetaAtEP = thePredState.theState().globalMomentum().theta();
 double phiAtEP   = thePredState.theState().globalMomentum().phi();
 double ptAtEP = thePredState.theState().globalMomentum().perp();
 double transverseCurvatureAtEP = field / ptAtEP*signTC;

 double x_v = thePredState.theState().globalPosition().x();
 double y_v = thePredState.theState().globalPosition().y();
 double z_v = thePredState.theState().globalPosition().z();
 double X = x_v - paramPt.x() - sin(phiAtEP) / transverseCurvatureAtEP;
 double Y = y_v - paramPt.y() + cos(phiAtEP) / transverseCurvatureAtEP;
 double SS = X*X + Y*Y;
 double S = sqrt(SS);
 
// The track parameters at the expansion point
  AlgebraicVector trackParameterFromExpansionPoint(6);
  trackParameterFromExpansionPoint[0] = transverseCurvatureAtEP;
  trackParameterFromExpansionPoint[1] = thetaAtEP;
  trackParameterFromExpansionPoint[3] = 1/transverseCurvatureAtEP  - signTC * S;
  
  trackParameterFromExpansionPoint[5] = part->currentState().mass();
  
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
		
  thePositionJacobian = AlgebraicMatrix(6,3,0);
  thePositionJacobian[2][0] = - Y / (SS);
  thePositionJacobian[2][1] = X / (SS);
  thePositionJacobian[3][0] = - signTC * X / S;
  thePositionJacobian[3][1] = - signTC * Y / S;
  thePositionJacobian[4][0] = thePositionJacobian[2][0]/tan(thetaAtEP)/transverseCurvatureAtEP;
  thePositionJacobian[4][1] = thePositionJacobian[2][1]/tan(thetaAtEP)/transverseCurvatureAtEP;
  thePositionJacobian[4][2] = 1;
 
//debug code - to be removed later 
//   cout<<"parameters for momentum jacobian: X "<<X<<endl;
//   cout<<"parameters for momentum jacobian: Y "<<Y<<endl;
//   cout<<"parameters for momentum jacobian: SS "<<SS<<endl;
//   cout<<"parameters for momentum jacobian: PhiAtEP "<<phiAtEP<<endl;
//   cout<<"parameters for momentum jacobian: curv "<<transverseCurvatureAtEP<<endl;
//   cout<<"sin phi Atep "<<sin(phiAtEP)<<endl;
//   cout<<"cos phi at EP "<<cos(phiAtEP)<<endl;
//   cout<<"upper  part is "<<X*cos(phiAtEP) + Y*sin(phiAtEP) <<endl;  
//   cout<<"lower part is"<<SS*transverseCurvatureAtEP*transverseCurvatureAtEP<<endl;

  theMomentumJacobian = AlgebraicMatrix(6,4,0);
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
				
					
  theMomentumJacobian[5][3] = 1;
  
// And finally the residuals:
  AlgebraicVector expansionPoint(3);
  expansionPoint[0] = thePredState.theState().globalPosition().x();
  expansionPoint[1] = thePredState.theState().globalPosition().y();
  expansionPoint[2] = thePredState.theState().globalPosition().z(); 
  AlgebraicVector momentumAtExpansionPoint(4);
  momentumAtExpansionPoint[0] = transverseCurvatureAtEP;  // Transverse Curv
  momentumAtExpansionPoint[1] = thetaAtEP;
  momentumAtExpansionPoint[2] = phiAtEP; 
  momentumAtExpansionPoint[3] = trackParameterFromExpansionPoint[5];
  theExpandedParams = trackParameterFromExpansionPoint;


  theConstantTerm = AlgebraicVector( trackParameterFromExpansionPoint -
  		  thePositionJacobian * expansionPoint -
  		  theMomentumJacobian * momentumAtExpansionPoint );
}
 
 
void ParticleKinematicLinearizedTrackState::computeNeutralJacobians() const
{
 GlobalPoint paramPt(theLinPoint);
 double thetaAtEP = thePredState.theState().globalMomentum().theta();
 double phiAtEP   = thePredState.theState().globalMomentum().phi();
 double ptAtEP = thePredState.theState().globalMomentum().perp();


 double x_v = thePredState.theState().globalPosition().x();
 double y_v = thePredState.theState().globalPosition().y();
 double z_v = thePredState.theState().globalPosition().z();
 double X = x_v - paramPt.x(); 
 double Y = y_v - paramPt.y();
  
// The track parameters at the expansion point
  AlgebraicVector trackParameterFromExpansionPoint(6);
  trackParameterFromExpansionPoint[0] = 1 / ptAtEP;
  trackParameterFromExpansionPoint[1] = thetaAtEP;
  trackParameterFromExpansionPoint[2] = phiAtEP;
  trackParameterFromExpansionPoint[3] = X*sin(phiAtEP) - Y*cos(phiAtEP);
  trackParameterFromExpansionPoint[4] = z_v - paramPt.z() - 
  	(X*cos(phiAtEP) + Y*sin(phiAtEP)) / tan(thetaAtEP);
  trackParameterFromExpansionPoint[5] = part->currentState().mass();
  
// The Jacobian: (all at the expansion point)
// [i,j]
// i = 0: rho = 1/pt , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
// j = 0: x_v, 1: y_v, 2: z_v
 thePositionJacobian = AlgebraicMatrix(6,3,0);
 thePositionJacobian[3][0] =   sin(phiAtEP);
 thePositionJacobian[3][1] = - cos(phiAtEP);
 thePositionJacobian[4][0] = - cos(phiAtEP)/tan(thetaAtEP);
 thePositionJacobian[4][1] = - sin(phiAtEP)/tan(thetaAtEP);
 thePositionJacobian[4][2] = 1;

// [i,j]
// i = 0: rho = 1/pt , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
// j = 0: rho = 1/pt , 1: theta, 2: phi_v
 theMomentumJacobian = AlgebraicMatrix(6,4,0); 
 theMomentumJacobian[0][0] = 1;
 theMomentumJacobian[1][1] = 1;
 theMomentumJacobian[2][2] = 1;

 theMomentumJacobian[3][2] = X*cos(phiAtEP) + Y*sin(phiAtEP); 
 theMomentumJacobian[4][1] = theMomentumJacobian[3][2]*
  	(1 + 1/(tan(thetaAtEP)*tan(thetaAtEP)));

 theMomentumJacobian[4][2] = (X*sin(phiAtEP) - Y*cos(phiAtEP))/tan(thetaAtEP);
 theMomentumJacobian[5][3] = 1;
 
// And finally the residuals:
 AlgebraicVector expansionPoint(3);
 expansionPoint[0] = thePredState.theState().globalPosition().x();
 expansionPoint[1] = thePredState.theState().globalPosition().y();
 expansionPoint[2] = thePredState.theState().globalPosition().z(); 
 AlgebraicVector momentumAtExpansionPoint(4);
 momentumAtExpansionPoint[0] = 1 / ptAtEP;
 momentumAtExpansionPoint[1] = thetaAtEP;
 momentumAtExpansionPoint[2] = phiAtEP; 
 momentumAtExpansionPoint[3] = trackParameterFromExpansionPoint[5];
 
 theExpandedParams = trackParameterFromExpansionPoint;
 theConstantTerm = AlgebraicVector( trackParameterFromExpansionPoint -
      		   thePositionJacobian * expansionPoint -
  		   theMomentumJacobian * momentumAtExpansionPoint );		   
}
     
reco::TransientTrack ParticleKinematicLinearizedTrackState::track() const
{
  throw VertexException(" ParticleKinematicLinearizedTrackState:: no TransientTrack to return");
}

 						   
						     
