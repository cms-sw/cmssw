#include "SiLinearChargeCollectionDrifter.h"

SiLinearChargeCollectionDrifter::SiLinearChargeCollectionDrifter(double dc,
								 double cdr,
								 double dv,
								 double av) :
  // Everything which does not depend on the specific det
  diffusionConstant(dc),
  chargeDistributionRMS(cdr),
  depletionVoltage(dv),
  appliedVoltage(av)
{
}

SiChargeCollectionDrifter::collection_type SiLinearChargeCollectionDrifter::drift(const SiChargeCollectionDrifter::ionization_type ion, 
										  const LocalVector& driftDir,double mt, double tn) {
  // prepare output
  collection_type _temp;
  _temp.resize(ion.size());
  // call the drift method for each deposit
  for (size_t i=0; i<ion.size(); i++){
    _temp[i] = drift(ion[i], driftDir, mt, tn);
  }
  return _temp;
}

SignalPoint SiLinearChargeCollectionDrifter::drift
(const EnergyDepositUnit& edu, const LocalVector& drift, double moduleThickness, double timeNormalisation) {
  // computes the fraction of the module the charge has to drift through,
  // ensuring it is bounded in [0,1]
  double depth = (moduleThickness/2.-edu.z());
  double thicknessFraction = depth/moduleThickness ; 
  thicknessFraction = thicknessFraction>0. ? thicknessFraction : 0. ;
  thicknessFraction = thicknessFraction<1. ? thicknessFraction : 1. ;
  
  // computes the drift time in the sensor
  double driftTime = -timeNormalisation*
    log(1.-2*depletionVoltage*thicknessFraction/
	(depletionVoltage+appliedVoltage))
    +chargeDistributionRMS;  
  
  // returns the signal: an energy on the surface, with a size due to diffusion.
  return SignalPoint(edu.x() + depth*drift.x()/drift.z(),
                     edu.y() + depth*drift.y()/drift.z(),
                     sqrt(2.*diffusionConstant*driftTime),
                     edu.energy());
}

