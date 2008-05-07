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
  
  //
  // Construct default classes
  //
  depletionVoltage       = conf_.getParameter<double>("DepletionVoltage");
  appliedVoltage         = conf_.getParameter<double>("AppliedVoltage");
  chargeMobility         = conf_.getParameter<double>("ChargeMobility");
  temperature            = conf_.getParameter<double>("Temperature");
  gevperelectron         =conf_.getParameter<double>("GevPerElectron");
  chargeDistributionRMS  =conf_.getParameter<double>("ChargeDistributionRMS");
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
SiHitDigitizer::processHit(const PSimHit& hit, const StripGeomDetUnit& det, GlobalVector bfield,float langle,
			   std::vector<double>& locAmpl, unsigned int& firstChannelWithSignal, unsigned int& lastChannelWithSignal){
  
  //
  // Compute the drift direction for this det
  //
  
  double moduleThickness = det.specificSurface().bounds().thickness(); // full detector thicness
  double timeNormalisation = (moduleThickness*moduleThickness)/(2.*depletionVoltage*chargeMobility);
  
  LocalVector driftDir = DriftDirection(&det,bfield,langle);
  
  //
  // Fully process one SimHit
  //
  
  SiChargeCollectionDrifter::ionization_type ion = theSiChargeDivider->divide(hit, driftDir, moduleThickness, det);
  
  theSiInduceChargeOnStrips->induce(theSiChargeCollectionDrifter->drift(ion,driftDir,moduleThickness,timeNormalisation),det,
				    locAmpl,firstChannelWithSignal,lastChannelWithSignal);
}

LocalVector SiHitDigitizer::DriftDirection(const StripGeomDetUnit* _detp,GlobalVector _bfield,float langle){
  // taken from ORCA/Tracker/SiStripDet/src/SiStripDet.cc
  Frame detFrame(_detp->surface().position(),_detp->surface().rotation());
  LocalVector Bfield=detFrame.toLocal(_bfield);
  
  if(langle==0.)
    edm::LogError("StripDigiInfo")<< "ERROR: Lorentz angle = 0 for module "<<_detp->geographicalId().rawId();
  
  float dir_x = -langle * Bfield.y();
  float dir_y = +langle * Bfield.x();
  float dir_z = 1.; // E field always in z direction
  LocalVector theDriftDirection = LocalVector(dir_x,dir_y,dir_z);
  if ( conf_.getUntrackedParameter<int>("VerbosityLevel") > 0 ) {
    edm::LogInfo("StripDigiInfo")<< " The drift direction in local coordinate is "<<theDriftDirection;
  }
  return theDriftDirection;
  
}
void SiHitDigitizer::setParticleDataTable(const ParticleDataTable * pdt)
{
  theSiChargeDivider->setParticleDataTable(pdt);
}
