#include "SimTracker/SiStripDigitizer/interface/SiHitDigitizer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeCollectionDrifter.h"
#include "SimTracker/SiStripDigitizer/interface/SiLinearChargeDivider.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "SimTracker/SiStripDigitizer/interface/SiTrivialInduceChargeOnStrips.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define CBOLTZ (1.38E-23)
#define e_SI (1.6E-19)

SiHitDigitizer::SiHitDigitizer(const edm::ParameterSet& conf,CLHEP::HepRandomEngine& eng ):conf_(conf),rndEngine(eng){
  
  // Construct default classes
  depletionVoltage       = conf_.getParameter<double>("DepletionVoltage");
  appliedVoltage         = conf_.getParameter<double>("AppliedVoltage");
  chargeMobility         = conf_.getParameter<double>("ChargeMobility");
  temperature            = conf_.getParameter<double>("Temperature");
  gevperelectron         = conf_.getParameter<double>("GevPerElectron");
  chargeDistributionRMS  = conf_.getParameter<double>("ChargeDistributionRMS");
  noDiffusion            = conf_.getParameter<bool>("noDiffusion");
  double diffusionConstant = CBOLTZ/e_SI * chargeMobility * temperature;
  if (noDiffusion) diffusionConstant *= 1.0e-3;
  
  theSiChargeDivider = new SiLinearChargeDivider(conf_,rndEngine);
  
  theSiChargeCollectionDrifter = 
    new SiLinearChargeCollectionDrifter(diffusionConstant,
					chargeDistributionRMS,
					depletionVoltage,
					appliedVoltage);

  theSiInduceChargeOnStrips = new SiTrivialInduceChargeOnStrips(conf,gevperelectron);
}

SiHitDigitizer::~SiHitDigitizer(){
  delete theSiChargeDivider;
  delete theSiChargeCollectionDrifter;
  delete theSiInduceChargeOnStrips;
}

void 
SiHitDigitizer::processHit(const PSimHit* hit, const StripGeomDetUnit& det, GlobalVector bfield,float langle,
			   std::vector<double>& locAmpl, size_t& firstChannelWithSignal, size_t& lastChannelWithSignal){
  
  // Compute the drift direction for this det
  double moduleThickness = det.specificSurface().bounds().thickness(); // active detector thicness
  double timeNormalisation = (moduleThickness*moduleThickness)/(2.*depletionVoltage*chargeMobility);
  LocalVector driftDir = DriftDirection(&det,bfield,langle);
  
  // Fully process one SimHit
  theSiInduceChargeOnStrips->induce(
      theSiChargeCollectionDrifter->drift(
          theSiChargeDivider->divide(hit, driftDir, moduleThickness, det),
          driftDir,moduleThickness,timeNormalisation),
      det,locAmpl,firstChannelWithSignal,lastChannelWithSignal);
}
