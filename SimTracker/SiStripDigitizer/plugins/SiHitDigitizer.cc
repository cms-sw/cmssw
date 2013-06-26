#include "SiHitDigitizer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SiLinearChargeCollectionDrifter.h"
#include "SiLinearChargeDivider.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "SiTrivialInduceChargeOnStrips.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define CBOLTZ (1.38E-23)
//#define e_SI (1.6E-19)
static const double CBOLTZ_over_e_SI = 1.38E-23/1.6E-19;
static const double noDiffusionMultiplier = 1.0e-3;

SiHitDigitizer::SiHitDigitizer(const edm::ParameterSet& conf, CLHEP::HepRandomEngine& eng) :
  depletionVoltage(conf.getParameter<double>("DepletionVoltage")),
  chargeMobility(conf.getParameter<double>("ChargeMobility")),
  theSiChargeDivider(new SiLinearChargeDivider(conf, eng)),
  theSiChargeCollectionDrifter(new SiLinearChargeCollectionDrifter(
         CBOLTZ_over_e_SI * chargeMobility * conf.getParameter<double>("Temperature") * (conf.getParameter<bool>("noDiffusion") ? noDiffusionMultiplier : 1.0),
         conf.getParameter<double>("ChargeDistributionRMS"),
         depletionVoltage,
         conf.getParameter<double>("AppliedVoltage"))),
  theSiInduceChargeOnStrips(new SiTrivialInduceChargeOnStrips(conf, conf.getParameter<double>("GevPerElectron"))) {
}

SiHitDigitizer::~SiHitDigitizer(){
}

void 
SiHitDigitizer::processHit(const PSimHit* hit, const StripGeomDetUnit& det, GlobalVector bfield,float langle,
			   std::vector<float>& locAmpl, size_t& firstChannelWithSignal, size_t& lastChannelWithSignal,
			   const TrackerTopology *tTopo){
  
  // Compute the drift direction for this det
  double moduleThickness = det.specificSurface().bounds().thickness(); // active detector thicness
  double timeNormalisation = (moduleThickness*moduleThickness)/(2.*depletionVoltage*chargeMobility);
  LocalVector driftDir = DriftDirection(&det,bfield,langle);
  
  // Fully process one SimHit
  theSiInduceChargeOnStrips->induce(
      theSiChargeCollectionDrifter->drift(
          theSiChargeDivider->divide(hit, driftDir, moduleThickness, det),
          driftDir,moduleThickness,timeNormalisation),
      det,locAmpl,firstChannelWithSignal,lastChannelWithSignal,tTopo);
}
