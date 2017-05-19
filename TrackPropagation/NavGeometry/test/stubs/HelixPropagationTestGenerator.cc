#include "TrackPropagation/NavGeometry/test/stubs/HelixPropagationTestGenerator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
//#include "Utilities/UI/interface/SimpleConfigurable.h"
//#include "CommonDet/DetUtilities/interface/DetExceptions.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"

#include <cfloat>

HelixPropagationTestGenerator::HelixPropagationTestGenerator(const MagneticField* field) :
  theField(field)
{
//   SimpleConfigurable<int> qMin_conf(-1,"PropagationTestGenerator:qMin");
//   qMin = qMin_conf.value();
//   SimpleConfigurable<int> qMax_conf(1,"PropagationTestGenerator:qMax"); 
//   qMax = qMax_conf.value();
//   SimpleConfigurable<float> ptMin_conf(0.9,"PropagationTestGenerator:ptMin");
//   ptMin = ptMin_conf.value();
//   SimpleConfigurable<float> ptMax_conf(1000.,"PropagationTestGenerator:ptMax");
//   ptMax = ptMax_conf.value();
//   SimpleConfigurable<bool> useLogPt_conf(0,"PropagationTestGenerator:useLogPt");
//   useLogPt = useLogPt_conf.value();

  qMin = -1;
  qMax = 1;
  ptMin = 0.9;
  ptMax = 1000.;
  useLogPt = false;
}

void HelixPropagationTestGenerator::setRangeCharge(const float min, const float max) {
  qMin = min;
  qMax = max;
}

void HelixPropagationTestGenerator::setRangePt(const float min, const float max) {
  ptMin = min;
  ptMax = max;
}

void HelixPropagationTestGenerator::generateStartValues () {

  //
  // generate random position (gaussian smearing)
  //
  currentPosition = VectorTypeExtended(posVx+sigVx*CLHEP::RandGauss::shoot(),
				       posVy+sigVy*CLHEP::RandGauss::shoot(),
				       posVz+sigVz*CLHEP::RandGauss::shoot());
  //
  // generate momentum vector from azimuthal angle, pseudo-
  // rapidity and transverse momentum
  //
  ExtendedDouble aPhi = CLHEP::RandFlat::shoot(phiMin,phiMax);
  ExtendedDouble aEta = CLHEP::RandFlat::shoot(etaMin,etaMax);
  ExtendedDouble aPt;
  if ( useLogPt )
    aPt = exp(CLHEP::RandFlat::shoot(log(ptMin>0?ptMin:FLT_MIN),log(ptMax)));
  else
    aPt = CLHEP::RandFlat::shoot(ptMin,ptMax);
  ExtendedDouble aTheta = 2.*atan(exp(-aEta));
  ExtendedDouble aP = sqrt((aPt*aPt)/(sin(aTheta)*sin(aTheta)));
  currentDirection = VectorTypeExtended(aP*sin(aTheta)*cos(aPhi),
					aP*sin(aTheta)*sin(aPhi),
					aP*cos(aTheta));
  //
  // generate charge (+/-1) and (signed) curvature
  //
  theCharge = CLHEP::RandFlat::shoot(qMin,qMax)<0. ? -1 : 1;
//    theCurvature = - theField->inInverseGeV(currentPosition.toPoint()).z()*theCharge/currentDirection.perp();
  theCurvature = - theField->inInverseGeV(GlobalPoint(currentPosition.x(),
						      currentPosition.y(),
						      currentPosition.z())).z()*
    theCharge/currentDirection.perp();
  //
  // set start state
  //
  setStartToCurrent();
}

GlobalPoint HelixPropagationTestGenerator::center() const {
  // check initialisation and return current position
  //if ( !initialised )  throw DetLogicError("HelixPropagationTestGenerator: attempt to use uninitialized helix");
//    return theCenter.toPoint();
  return GlobalPoint(theCenter.x(),
		     theCenter.y(),
		     theCenter.z());
}

void HelixPropagationTestGenerator::setStart (const GlobalPoint& position,
					      const GlobalVector& momentum) {
  //
  // reset current state
  //
  currentPosition = VectorTypeExtended(position.x(),
				       position.y(),
				       position.z());
  currentDirection = VectorTypeExtended(momentum.x(),
					momentum.y(),
					momentum.z());
  //
  // reset curvature
  //
  theCurvature = - theField->inInverseGeV(position).z()*
    theCharge/currentDirection.perp();
  //
  // set start state
  //
  setStartToCurrent();
}

void HelixPropagationTestGenerator::setStartToCurrent() {
  //
  // set current state
  //
  startPosition = currentPosition;
  startDirection = currentDirection;
  //
  // get center of helix
  //
  ExtendedDouble rscale = 1./theCurvature/currentDirection.perp();
  theCenter = VectorTypeExtended(currentPosition.x()-currentDirection.y()*rscale,
				 currentPosition.y()+currentDirection.x()*rscale,
				 currentPosition.z());
  //
  // set total path length and angle within helix
  //
  sTotal = 0;
  startPhiHelix = startDirection.phi();
  startPhiHelix += theCharge>0 ? M_PI/2. : -M_PI/2.;
//    cout << "Helix center at " << theCenter << endl;
//    cout << "start helix angle is " << startPhiHelix << endl;
//    cout << "q = " << theCharge << ", 1/curvature = " << 1/theCurvature << endl;
  initialised = true;
}

void HelixPropagationTestGenerator::setStartOppositeToCurrent() {
  //
  currentDirection *= -1;
  theCharge *= -1;
  theCurvature *= -1;
  setStartToCurrent();
}

HelixPropagationTestGenerator::ExtendedDouble 
HelixPropagationTestGenerator::bidirectionalStep (const ExtendedDouble stepSize) {
  //
  // update total path length, calculate transversal path length, change in 
  // angle and new angle within the helix
  //
//  cout << "Propagation for step " << stepSize << endl;
  sTotal += stepSize;
//  cout << "New total path = " << sTotal << endl;
  ExtendedDouble sTransversal = sTotal*startDirection.perp()/startDirection.mag();
//  cout << "New transversal path = " << sTransversal << endl;
  ExtendedDouble dPhi = sTransversal*theCurvature;
//  cout << "deltaPhi = " << dPhi << endl;
  ExtendedDouble phiHelix = startPhiHelix + dPhi;
//  cout << "phiHelix = " << phiHelix << endl;
  //
  // recalculate position and direction in double precision
  //
  currentPosition = VectorTypeExtended(theCenter.x()+cos(phiHelix)/std::abs(theCurvature),
				       theCenter.y()+sin(phiHelix)/std::abs(theCurvature),
				       theCenter.z()+dPhi/theCurvature*startDirection.z()/startDirection.perp());
  currentDirection = VectorTypeExtended(cos(startDirection.phi()+dPhi)*startDirection.perp(),
					sin(startDirection.phi()+dPhi)*startDirection.perp(),
					startDirection.z());
  //
  // return total path length
  //
  return sTotal;
}
