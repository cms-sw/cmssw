#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeCollectionDrifter.h"



SiLinearChargeCollectionDrifter::SiLinearChargeCollectionDrifter(double dc,
								 double cdr,
								 double dv,
								 double av){
  //
  // Everything which does not depend on the specific det
  //
  diffusionConstant = dc;
  chargeDistributionRMS = cdr;
  depletionVoltage = dv;
  appliedVoltage = av;
}

SiChargeCollectionDrifter::collection_type SiLinearChargeCollectionDrifter::drift(const SiChargeCollectionDrifter::ionization_type ion, 
										  const LocalVector& driftDir,double mt, double tn){

  moduleThickness = mt;
  timeNormalisation = tn;
  
  collection_type _temp;
  _temp.resize(ion.size());
  
  for (unsigned int i=0; i<ion.size(); i++){
    _temp[i] = drift(ion[i], driftDir);
  }
  
  return _temp;

}

SignalPoint SiLinearChargeCollectionDrifter::drift
(const EnergyDepositUnit& edu, const LocalVector& drift){
  
  double tanLorentzAngleX = drift.x()/drift.z();
  double tanLorentzAngleY = drift.y()/drift.z();
  
  double segX = (edu).x();
  double segY = (edu).y();
  double segZ = (edu).z();
  
  double thicknessFraction = (moduleThickness/2.-segZ)/moduleThickness ; 
  // fix the bug due to  rounding on entry and exit point
  thicknessFraction = thicknessFraction>0. ? thicknessFraction : 0. ;
  thicknessFraction = thicknessFraction<1. ? thicknessFraction : 1. ;
  
  double driftTime = -timeNormalisation*
    log(1.-2*depletionVoltage*thicknessFraction/
	(depletionVoltage+appliedVoltage))
    +chargeDistributionRMS;  
  
  double sigma = sqrt(2.*diffusionConstant*driftTime);

  double xDriftDueToMagField // Drift along X due to BField
    = (moduleThickness/2. - segZ)*tanLorentzAngleX;
  double yDriftDueToMagField // Drift along Y due to BField
    = (moduleThickness/2. - segZ)*tanLorentzAngleY;
  double positionX = segX + xDriftDueToMagField;
  double positionY = segY + yDriftDueToMagField;
  
  return SignalPoint(positionX,positionY,sigma,
		     (edu).energy());  
}
			
