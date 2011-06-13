#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeCollectionDrifter.h"

SiLinearChargeCollectionDrifter::SiLinearChargeCollectionDrifter(double dc,
								 double cdr,
								 double dv,
								 double av){
  // Everything which does not depend on the specific det
  diffusionConstant = dc;
  chargeDistributionRMS = cdr;
  depletionVoltage = dv;
  appliedVoltage = av;
  dVOndVaV = 2.*depletionVoltage/(depletionVoltage+appliedVoltage);
}

SiChargeCollectionDrifter::collection_type* SiLinearChargeCollectionDrifter::drift(const SiChargeCollectionDrifter::ionization_type* ion, 
										  const LocalVector& driftDir,double mt, double tn) {
  // set some variables used in the main method
  moduleThickness = mt;
  timeNormalisation = tn;
  dxOnDz = driftDir.x()/driftDir.z();
  dyOnDz = driftDir.y()/driftDir.z();
  // prepare output
  size_t ionsize = ion->size();

  m_temp.resize(ionsize);
  // call the drift method for each deposit
  for (size_t i=0; i<ionsize; i++){
    m_temp[i] = drift((*ion)[i]);
  }
  return &m_temp;
}

SignalPoint SiLinearChargeCollectionDrifter::drift
(const EnergyDepositUnit& edu) {
  // computes the fraction of the module the charge has to drift through,
  // ensuring it is bounded in [0,1]
  double depth = (moduleThickness/2.-edu.z());
  double thicknessFraction = depth/moduleThickness ; 
  thicknessFraction = thicknessFraction>0. ? thicknessFraction : 0. ;
  thicknessFraction = thicknessFraction<1. ? thicknessFraction : 1. ;
  
  // computes the drift time in the sensor
  double driftTime = -timeNormalisation*log(1.-dVOndVaV*thicknessFraction)
    +chargeDistributionRMS;  
  
  // returns the signal: an energy on the surface, with a size due to diffusion.
  return SignalPoint(edu.x() + depth*dxOnDz,
                     edu.y() + depth*dyOnDz,
                     sqrt(2.*diffusionConstant*driftTime),
                     edu.energy());
}

