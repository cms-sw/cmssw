#ifndef Tracker_SiLinearChargeCollectionDrifter_H
#define Tracker_SiLinearChargeCollectionDrifter_H

#include "SiChargeCollectionDrifter.h"
#include "EnergyDepositUnit.h"
/**
 * Concrete implementation of SiChargeCollectionDrifter. Drifts the charges linearly.
 * Drift each energy deposits in the bulk to the surface.                                                              
 * The resulting position depends on the Lorentz angle, and a sigma is computed                                        
 * that describes the diffusion.               
 */
class SiLinearChargeCollectionDrifter : public SiChargeCollectionDrifter{
 public:
  SiLinearChargeCollectionDrifter(double,double,double,double);
  SiChargeCollectionDrifter::collection_type drift(const SiChargeCollectionDrifter::ionization_type, 
                                                   const LocalVector&,double,double);
 private:
  SignalPoint drift(const EnergyDepositUnit&, const LocalVector&,double,double);
 private:
  const double diffusionConstant;
  const double chargeDistributionRMS;
  const double depletionVoltage;
  const double appliedVoltage;
};
#endif

