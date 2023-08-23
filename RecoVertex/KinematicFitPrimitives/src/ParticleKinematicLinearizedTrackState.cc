#include "RecoVertex/KinematicFitPrimitives/interface/ParticleKinematicLinearizedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicRefittedTrackState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicPerigeeConversions.h"

const AlgebraicVector6& ParticleKinematicLinearizedTrackState::constantTerm() const {
  if (!jacobiansAvailable)
    computeJacobians();
  return theConstantTerm;
}

const AlgebraicMatrix63& ParticleKinematicLinearizedTrackState::positionJacobian() const {
  if (!jacobiansAvailable)
    computeJacobians();
  return thePositionJacobian;
}

const AlgebraicMatrix64& ParticleKinematicLinearizedTrackState::momentumJacobian() const {
  if (!jacobiansAvailable)
    computeJacobians();
  return theMomentumJacobian;
}

const AlgebraicVector6& ParticleKinematicLinearizedTrackState::parametersFromExpansion() const {
  if (!jacobiansAvailable)
    computeJacobians();
  return theExpandedParams;
}

AlgebraicVector6 ParticleKinematicLinearizedTrackState::predictedStateParameters() const {
  if (!jacobiansAvailable)
    computeJacobians();
  return thePredState.perigeeParameters().vector();
}

AlgebraicSymMatrix66 ParticleKinematicLinearizedTrackState::predictedStateWeight(int& error) const {
  if (!jacobiansAvailable)
    computeJacobians();
  int i = 0;
  AlgebraicSymMatrix66 z = thePredState.perigeeError().weightMatrix(i);
  error = i;
  return z;
  //   return thePredState.perigeeError().weightMatrix(error);
}

AlgebraicSymMatrix66 ParticleKinematicLinearizedTrackState::predictedStateError() const {
  if (!jacobiansAvailable)
    computeJacobians();
  return thePredState.perigeeError().covarianceMatrix();
}

// ImpactPointMeasurement  ParticleKinematicLinearizedTrackState::impactPointMeasurement() const
// { throw VertexException(" ParticleKinematicLinearizedTrackState::impact point measurement is not implemented for kinematic classes!");}

TrackCharge ParticleKinematicLinearizedTrackState::charge() const { return part->initialState().particleCharge(); }

RefCountedKinematicParticle ParticleKinematicLinearizedTrackState::particle() const { return part; }

bool ParticleKinematicLinearizedTrackState::operator==(LinearizedTrackState<6>& other) const {
  const ParticleKinematicLinearizedTrackState* otherP =
      dynamic_cast<const ParticleKinematicLinearizedTrackState*>(&other);
  if (otherP == nullptr) {
    throw VertexException(
        " ParticleKinematicLinearizedTrackState:: don't know how to compare myself to non-kinematic track state");
  }
  return (*(otherP->particle()) == *part);
}

bool ParticleKinematicLinearizedTrackState::hasError() const {
  if (!jacobiansAvailable)
    computeJacobians();
  return thePredState.isValid();
}

// here we make a and b matrices of
// our measurement function expansion.
// marices will be almost the same as for
// classical perigee, but bigger:
// (6x3) and (6x4) respectivelly.
void ParticleKinematicLinearizedTrackState::computeJacobians() const {
  GlobalPoint paramPt(theLinPoint);
  thePredState = builder(part->currentState(), paramPt);
  //  bool valid = thePredState.isValid();
  //  if (!valid) std::cout <<"Help!!!!!!!!! State is invalid\n";
  //  if (!valid) return;
  if (std::abs(theCharge) < 1e-5) {
    //neutral track
    computeNeutralJacobians();
  } else {
    //charged track
    computeChargedJacobians();
  }
  jacobiansAvailable = true;
}

ReferenceCountingPointer<LinearizedTrackState<6> >
ParticleKinematicLinearizedTrackState::stateWithNewLinearizationPoint(const GlobalPoint& newLP) const {
  RefCountedKinematicParticle pr = part;
  return new ParticleKinematicLinearizedTrackState(newLP, pr);
}

ParticleKinematicLinearizedTrackState::RefCountedRefittedTrackState
ParticleKinematicLinearizedTrackState::createRefittedTrackState(const GlobalPoint& vertexPosition,
                                                                const AlgebraicVector4& vectorParameters,
                                                                const AlgebraicSymMatrix77& covarianceMatrix) const {
  KinematicPerigeeConversions conversions;
  KinematicState lst = conversions.kinematicState(
      vectorParameters, vertexPosition, charge(), covarianceMatrix, part->currentState().magneticField());
  RefCountedRefittedTrackState rst =
      RefCountedRefittedTrackState(new KinematicRefittedTrackState(lst, vectorParameters));
  return rst;
}

AlgebraicVector4 ParticleKinematicLinearizedTrackState::predictedStateMomentumParameters() const {
  if (!jacobiansAvailable)
    computeJacobians();
  AlgebraicVector4 res;
  res[0] = thePredState.perigeeParameters().vector()(0);
  res[1] = thePredState.perigeeParameters().vector()(1);
  res[2] = thePredState.perigeeParameters().vector()(2);
  res[3] = thePredState.perigeeParameters().vector()(5);
  return res;
}

AlgebraicSymMatrix44 ParticleKinematicLinearizedTrackState::predictedStateMomentumError() const {
  int error;
  if (!jacobiansAvailable)
    computeJacobians();
  AlgebraicSymMatrix44 res;
  AlgebraicSymMatrix33 m3 = thePredState.perigeeError().weightMatrix(error).Sub<AlgebraicSymMatrix33>(0, 0);
  res.Place_at(m3, 0, 0);
  res(3, 0) = thePredState.perigeeError().weightMatrix(error)(5, 5);
  res(3, 1) = thePredState.perigeeError().weightMatrix(error)(5, 0);
  res(3, 2) = thePredState.perigeeError().weightMatrix(error)(5, 1);
  res(3, 3) = thePredState.perigeeError().weightMatrix(error)(5, 2);
  return res;
}

double ParticleKinematicLinearizedTrackState::weightInMixture() const { return 1.; }

std::vector<ReferenceCountingPointer<LinearizedTrackState<6> > > ParticleKinematicLinearizedTrackState::components()
    const {
  std::vector<ReferenceCountingPointer<LinearizedTrackState<6> > > res;
  res.reserve(1);
  res.push_back(RefCountedLinearizedTrackState(const_cast<ParticleKinematicLinearizedTrackState*>(this)));
  return res;
}

AlgebraicVector6 ParticleKinematicLinearizedTrackState::refittedParamFromEquation(
    const RefCountedRefittedTrackState& theRefittedState) const {
  AlgebraicVectorM momentum = theRefittedState->momentumVector();
  if ((momentum(2) * predictedStateMomentumParameters()(2) < 0) && (std::fabs(momentum(2)) > M_PI / 2)) {
    if (predictedStateMomentumParameters()(2) < 0.)
      momentum(2) -= 2 * M_PI;
    if (predictedStateMomentumParameters()(2) > 0.)
      momentum(2) += 2 * M_PI;
  }

  AlgebraicVector3 vertexPosition;
  vertexPosition(0) = theRefittedState->position().x();
  vertexPosition(1) = theRefittedState->position().y();
  vertexPosition(2) = theRefittedState->position().z();
  AlgebraicVector6 param = constantTerm() + positionJacobian() * vertexPosition + momentumJacobian() * momentum;

  if (param(2) > M_PI)
    param(2) -= 2 * M_PI;
  if (param(2) < -M_PI)
    param(2) += 2 * M_PI;

  return param;
}

void ParticleKinematicLinearizedTrackState::checkParameters(AlgebraicVector6& parameters) const {
  if (parameters(2) > M_PI)
    parameters(2) -= 2 * M_PI;
  if (parameters(2) < -M_PI)
    parameters(2) += 2 * M_PI;
}

void ParticleKinematicLinearizedTrackState::computeChargedJacobians() const {
  GlobalPoint paramPt(theLinPoint);

  double field = part->currentState().magneticField()->inInverseGeV(thePredState.theState().globalPosition()).z();
  double signTC = -part->currentState().particleCharge();

  double thetaAtEP = thePredState.theState().globalMomentum().theta();
  double phiAtEP = thePredState.theState().globalMomentum().phi();
  double ptAtEP = thePredState.theState().globalMomentum().perp();
  double transverseCurvatureAtEP = field / ptAtEP * signTC;

  // Fix calculation for case where magnetic field swaps sign between previous state and current state
  if (field * part->currentState().magneticField()->inInverseGeV(part->currentState().globalPosition()).z() < 0.) {
    signTC = -signTC;
  }

  double x_v = thePredState.theState().globalPosition().x();
  double y_v = thePredState.theState().globalPosition().y();
  double z_v = thePredState.theState().globalPosition().z();
  double X = x_v - paramPt.x() - sin(phiAtEP) / transverseCurvatureAtEP;
  double Y = y_v - paramPt.y() + cos(phiAtEP) / transverseCurvatureAtEP;
  double SS = X * X + Y * Y;
  double S = sqrt(SS);

  // The track parameters at the expansion point
  theExpandedParams[0] = transverseCurvatureAtEP;
  theExpandedParams[1] = thetaAtEP;
  theExpandedParams[3] = 1 / transverseCurvatureAtEP - signTC * S;

  theExpandedParams[5] = part->currentState().mass();

  double phiFEP;
  if (std::abs(X) > std::abs(Y)) {
    double signX = (X > 0.0 ? +1.0 : -1.0);
    phiFEP = -signTC * signX * acos(signTC * Y / S);
  } else {
    phiFEP = asin(-signTC * X / S);
    if ((signTC * Y) < 0.0)
      phiFEP = M_PI - phiFEP;
  }
  if (phiFEP > M_PI)
    phiFEP -= 2 * M_PI;
  theExpandedParams[2] = phiFEP;
  theExpandedParams[4] =
      z_v - paramPt.z() - (phiAtEP - theExpandedParams[2]) / tan(thetaAtEP) / transverseCurvatureAtEP;

  thePositionJacobian(2, 0) = -Y / (SS);
  thePositionJacobian(2, 1) = X / (SS);
  thePositionJacobian(3, 0) = -signTC * X / S;
  thePositionJacobian(3, 1) = -signTC * Y / S;
  thePositionJacobian(4, 0) = thePositionJacobian(2, 0) / tan(thetaAtEP) / transverseCurvatureAtEP;
  thePositionJacobian(4, 1) = thePositionJacobian(2, 1) / tan(thetaAtEP) / transverseCurvatureAtEP;
  thePositionJacobian(4, 2) = 1;

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

  theMomentumJacobian(0, 0) = 1;
  theMomentumJacobian(1, 1) = 1;

  theMomentumJacobian(2, 0) =
      -(X * cos(phiAtEP) + Y * sin(phiAtEP)) / (SS * transverseCurvatureAtEP * transverseCurvatureAtEP);

  theMomentumJacobian(2, 2) = (Y * cos(phiAtEP) - X * sin(phiAtEP)) / (SS * transverseCurvatureAtEP);

  theMomentumJacobian(3, 0) =
      (signTC * (Y * cos(phiAtEP) - X * sin(phiAtEP)) / S - 1) / (transverseCurvatureAtEP * transverseCurvatureAtEP);

  theMomentumJacobian(3, 2) = signTC * (X * cos(phiAtEP) + Y * sin(phiAtEP)) / (S * transverseCurvatureAtEP);

  theMomentumJacobian(4, 0) =
      (phiAtEP - theExpandedParams(2)) / tan(thetaAtEP) / (transverseCurvatureAtEP * transverseCurvatureAtEP) +
      theMomentumJacobian(2, 0) / tan(thetaAtEP) / transverseCurvatureAtEP;

  theMomentumJacobian(4, 1) =
      (phiAtEP - theExpandedParams(2)) * (1 + 1 / (tan(thetaAtEP) * tan(thetaAtEP))) / transverseCurvatureAtEP;

  theMomentumJacobian(4, 2) = (theMomentumJacobian(2, 2) - 1) / tan(thetaAtEP) / transverseCurvatureAtEP;

  theMomentumJacobian(5, 3) = 1;

  // And finally the residuals:
  AlgebraicVector3 expansionPoint;
  expansionPoint[0] = thePredState.theState().globalPosition().x();
  expansionPoint[1] = thePredState.theState().globalPosition().y();
  expansionPoint[2] = thePredState.theState().globalPosition().z();
  AlgebraicVector4 momentumAtExpansionPoint;
  momentumAtExpansionPoint[0] = transverseCurvatureAtEP;  // Transverse Curv
  momentumAtExpansionPoint[1] = thetaAtEP;
  momentumAtExpansionPoint[2] = phiAtEP;
  momentumAtExpansionPoint[3] = theExpandedParams[5];

  theConstantTerm = AlgebraicVector6(theExpandedParams - thePositionJacobian * expansionPoint -
                                     theMomentumJacobian * momentumAtExpansionPoint);
}

void ParticleKinematicLinearizedTrackState::computeNeutralJacobians() const {
  GlobalPoint paramPt(theLinPoint);
  double thetaAtEP = thePredState.theState().globalMomentum().theta();
  double phiAtEP = thePredState.theState().globalMomentum().phi();
  double ptAtEP = thePredState.theState().globalMomentum().perp();

  double x_v = thePredState.theState().globalPosition().x();
  double y_v = thePredState.theState().globalPosition().y();
  double z_v = thePredState.theState().globalPosition().z();
  double X = x_v - paramPt.x();
  double Y = y_v - paramPt.y();

  // The track parameters at the expansion point
  theExpandedParams[0] = 1 / ptAtEP;
  theExpandedParams[1] = thetaAtEP;
  theExpandedParams[2] = phiAtEP;
  theExpandedParams[3] = X * sin(phiAtEP) - Y * cos(phiAtEP);
  theExpandedParams[4] = z_v - paramPt.z() - (X * cos(phiAtEP) + Y * sin(phiAtEP)) / tan(thetaAtEP);
  theExpandedParams[5] = part->currentState().mass();

  // The Jacobian: (all at the expansion point)
  // [i,j]
  // i = 0: rho = 1/pt , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
  // j = 0: x_v, 1: y_v, 2: z_v
  thePositionJacobian(3, 0) = sin(phiAtEP);
  thePositionJacobian(3, 1) = -cos(phiAtEP);
  thePositionJacobian(4, 0) = -cos(phiAtEP) / tan(thetaAtEP);
  thePositionJacobian(4, 1) = -sin(phiAtEP) / tan(thetaAtEP);
  thePositionJacobian(4, 2) = 1;

  // [i,j]
  // i = 0: rho = 1/pt , 1: theta, 2: phi_p, 3: epsilon, 4: z_p
  // j = 0: rho = 1/pt , 1: theta, 2: phi_v
  theMomentumJacobian(0, 0) = 1;
  theMomentumJacobian(1, 1) = 1;
  theMomentumJacobian(2, 2) = 1;

  theMomentumJacobian(3, 2) = X * cos(phiAtEP) + Y * sin(phiAtEP);
  theMomentumJacobian(4, 1) = theMomentumJacobian(3, 2) * (1 + 1 / (tan(thetaAtEP) * tan(thetaAtEP)));

  theMomentumJacobian(4, 2) = (X * sin(phiAtEP) - Y * cos(phiAtEP)) / tan(thetaAtEP);
  theMomentumJacobian(5, 3) = 1;

  // And finally the residuals:
  AlgebraicVector3 expansionPoint;
  expansionPoint[0] = thePredState.theState().globalPosition().x();
  expansionPoint[1] = thePredState.theState().globalPosition().y();
  expansionPoint[2] = thePredState.theState().globalPosition().z();
  AlgebraicVector4 momentumAtExpansionPoint;
  momentumAtExpansionPoint[0] = 1 / ptAtEP;
  momentumAtExpansionPoint[1] = thetaAtEP;
  momentumAtExpansionPoint[2] = phiAtEP;
  momentumAtExpansionPoint[3] = theExpandedParams[5];

  theConstantTerm = AlgebraicVector6(theExpandedParams - thePositionJacobian * expansionPoint -
                                     theMomentumJacobian * momentumAtExpansionPoint);
}

reco::TransientTrack ParticleKinematicLinearizedTrackState::track() const {
  throw VertexException(" ParticleKinematicLinearizedTrackState:: no TransientTrack to return");
}
