#include "SimTracker/SiStripDigitizer/interface/SiHitDigitizer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeCollectionDrifter.h"
#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeDivider.h"

#include "Geometry/Vector/interface/LocalPoint.h"
#include "SimTracker/SiStripDigitizer/interface/SiTrivialInduceChargeOnStrips.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#define CBOLTZ (1.38E-23)
#define e_SI (1.6E-19)

SiHitDigitizer::SiHitDigitizer(const edm::ParameterSet const& conf, const StripGeomDetUnit *det):conf_(conf){

  //
  // Construct default classes
  //
  
  theSiChargeDivider = new SiLinearChargeDivider();
  
  depletionVoltage=conf_.getParameter<double>("DepletionVoltage");
  appliedVoltage=conf_.getParameter<double>("AppliedVoltage");
  chargeMobility=conf_.getParameter<double>("ChargeMobility");
  temperature=conf_.getParameter<double>("Temperature");
  double diffusionConstant = CBOLTZ/e_SI * chargeMobility * temperature;
  noDiffusion=conf_.getParameter<bool>("noDiffusion");
  if (noDiffusion) diffusionConstant *= 1.0e-3;
  chargeDistributionRMS=conf_.getParameter<double>("ChargeDistributionRMS");
  double moduleThickness = det->specificSurface().bounds().thickness(); // full detector thicness
  //  double moduleThickness = type.bounds().thickness();
  // If no diffusion requested, don't set it quite to zero to avoid
  // divide by zero errors.
  double timeNormalisation = pow(moduleThickness,2)/(2.*depletionVoltage*chargeMobility);
 
  theSiChargeCollectionDrifter = 
    new SiLinearChargeCollectionDrifter(moduleThickness,
					timeNormalisation,
					diffusionConstant,
					temperature,
					chargeDistributionRMS,
					depletionVoltage,
					appliedVoltage);

  gevperelectron=conf_.getParameter<double>("GevPerElectron");
 
  theSiInduceChargeOnStrips = new SiTrivialInduceChargeOnStrips(gevperelectron);
}

SiHitDigitizer::hit_map_type SiHitDigitizer::processHit(const PSimHit& hit, const StripGeomDetUnit& det){

  //
  // Fully process one SimHit
  //
  SiChargeCollectionDrifter::ionization_type ion = theSiChargeDivider->divide(hit, det);
  
  //
  // Compute the drift direction for this det
  //
  
  LocalPoint centre(0.,0.);
  //  LocalVector drift = det.driftDirection(centre); // aggiungere dopo! (AG)
  LocalVector drift; // temporary! (AG)
  
  return theSiInduceChargeOnStrips->induce(
					   theSiChargeCollectionDrifter->drift(ion,drift),
					   det);
}
