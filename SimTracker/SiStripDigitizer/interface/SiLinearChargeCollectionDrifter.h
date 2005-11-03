#ifndef Tracker_SiLinearChargeCollectionDrifter_H
#define Tracker_SiLinearChargeCollectionDrifter_H

#include "SimTracker/SiStripDigitizer/interface/SiChargeCollectionDrifter.h"
#include "SimTracker/SiStripDigitizer/interface/EnergyDepositUnit.h"
/**
 * Concrete implementation of SiChargeCollectionDrifter. Drifts the charges linearly.
 */
class SiLinearChargeCollectionDrifter : public SiChargeCollectionDrifter{
 public:
  SiLinearChargeCollectionDrifter(double,double,double,double,double,double,double);
  SiChargeCollectionDrifter::collection_type drift (const SiChargeCollectionDrifter::ionization_type, const LocalVector&);

 private:
  
  SignalPoint drift
    (const EnergyDepositUnit&, const LocalVector&);
  
  double moduleThickness;
  double timeNormalisation;
  double diffusionConstant;
  double temperature;
  double chargeDistributionRMS;
  double depletionVoltage;
  double appliedVoltage;
  
};


#endif

