#include "TrackPropagation/NavGeometry/test/stubs/PropagationTestGenerator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"
//#include "CommonDet/DetUtilities/interface/DetExceptions.h"
#include "TrackingTools/GeomPropagators/interface/PropagationExceptions.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

PropagationTestGenerator::PropagationTestGenerator() {
  initialised = false;
//   SimpleConfigurable<float> phiMin_conf(0.,"PropagationTestGenerator:phiMin");
//   phiMin = phiMin_conf.value();
//   SimpleConfigurable<float> phiMax_conf(2*M_PI,"PropagationTestGenerator:phiMax");
//   phiMax = phiMax_conf.value();
//   SimpleConfigurable<float> etaMin_conf(-2.5,"PropagationTestGenerator:etaMin");
//   etaMin = etaMin_conf.value();
//   SimpleConfigurable<float> etaMax_conf(2.5,"PropagationTestGenerator:etaMax");
//   etaMax = etaMax_conf.value();
//   SimpleConfigurable<float> posVx_conf(0.,"PropagationTestGenerator:posVx");
//   posVx = posVx_conf.value();
//   SimpleConfigurable<float> posVy_conf(0.,"PropagationTestGenerator:posVy");
//   posVy = posVy_conf.value();
//   SimpleConfigurable<float> posVz_conf(0.,"PropagationTestGenerator:posVz");
//   posVz = posVz_conf.value();
//   SimpleConfigurable<float> sigVx_conf(0.0010,"PropagationTestGenerator:sigVx");
//   sigVx = sigVx_conf.value();
//   SimpleConfigurable<float> sigVy_conf(0.0010,"PropagationTestGenerator:sigVy");
//   sigVy = sigVy_conf.value();
//   SimpleConfigurable<float> sigVz_conf(10.,"PropagationTestGenerator:sigVz");
//   sigVz = sigVz_conf.value();
//   SimpleConfigurable<bool> useLogStep_conf(0,"PropagationTestGenerator:useLogStep");
//   useLogStep = useLogStep_conf.value();

  phiMin = 0.;
  phiMax = 2*M_PI;
  etaMin = -2.5;
  etaMax = 2.5;
  posVx = 0.;
  posVy = 0.;
  posVz = 0.;
  sigVx = 0.0010;
  sigVy = 0.0010;
  sigVz = 10.;
  useLogStep = 0;
}

void PropagationTestGenerator::setRangePhi(const float min, const float max) {
  phiMin = min;
  phiMax = max;
}

void PropagationTestGenerator::setRangeEta(const float min, const float max) {
  etaMin = min;
  etaMax = max;
}

void PropagationTestGenerator::setVertexSmearing(const float x, const float y, const float z) {
  sigVx = x;
  sigVy = y;
  sigVz = z;
}

int PropagationTestGenerator::charge() const {
  // check initialisation and return current direction
  if ( !initialised )  throw PropagationException("HelixPropagationTestGenerator: attempt to use uninitialized helix");
  return theCharge;
}

PropagationTestGenerator::ExtendedDouble 
PropagationTestGenerator::transverseCurvature() const {
  // check initialisation and return current position
  if ( !initialised )  throw PropagationException("HelixPropagationTestGenerator: attempt to use uninitialized helix");
  return theCurvature;
}

void PropagationTestGenerator::generateStartValues () {

  //
  // generate random position (gaussian smearing)
  //
  currentPosition = VectorTypeExtended(sigVx*CLHEP::RandGauss::shoot(),
				       sigVy*CLHEP::RandGauss::shoot(),
				       sigVz*CLHEP::RandGauss::shoot());
  //
  // generate momentum vector from azimuthal angle, pseudo-
  // rapidity and transverse momentum
  //
  ExtendedDouble aPhi = CLHEP::RandFlat::shoot(phiMin,phiMax);
  ExtendedDouble aEta = CLHEP::RandFlat::shoot(etaMin,etaMax);
  ExtendedDouble aTheta = 2.*atan(exp(-aEta));
  currentDirection = VectorTypeExtended(sin(aTheta)*cos(aPhi),
					sin(aTheta)*sin(aPhi),
					cos(aTheta));
  //
  // set start state
  //
  setStartToCurrent();
}

PropagationTestGenerator::ExtendedDouble 
PropagationTestGenerator::randomStepForward(const float maxStep) {
  // check initialisation and do a random step (>0)
  if ( !initialised )  throw PropagationException("PropagationTestGenerator: attempt to use uninitialized trajectory");
  ExtendedDouble step;
  if ( useLogStep )
    step = exp(CLHEP::RandFlat::shoot(log(1.e-5),log(maxStep)));
  else
    step = CLHEP::RandFlat::shoot(0.,maxStep);
  return bidirectionalStep(step);
}

PropagationTestGenerator::ExtendedDouble 
PropagationTestGenerator::randomStepBackward(const float maxStep) {
  // check initialisation and do a random step (<0)
  if ( !initialised )  throw PropagationException("PropagationTestGenerator: attempt to use uninitialized trajectory");
  ExtendedDouble step;
  if ( useLogStep )
    step = exp(CLHEP::RandFlat::shoot(log(1.e-5),maxStep));
  else
    step = CLHEP::RandFlat::shoot(0.,maxStep);
  return bidirectionalStep(-step);
}

GlobalPoint PropagationTestGenerator::position() const {
  // check initialisation and return current position
  if ( !initialised )  throw PropagationException("PropagationTestGenerator: attempt to use uninitialized trajectory");
//    return currentPosition.toPoint();
  return GlobalPoint(currentPosition.x(),
		     currentPosition.y(),
		     currentPosition.z());
}

GlobalVector PropagationTestGenerator::momentum() const {
  // check initialisation and return current direction
  if ( !initialised )  throw PropagationException("PropagationTestGenerator: attempt to use uninitialized trajectory");
//    return currentDirection.toVector();
  return GlobalVector(currentDirection.x(),
		      currentDirection.y(),
		      currentDirection.z());
}

