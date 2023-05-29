#include <iostream>
#include <cmath>

#include "SimTracker/SiPhase2Digitizer/plugins/PSPDigitizerAlgorithm.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

using namespace edm;

void PSPDigitizerAlgorithm::init(const edm::EventSetup& es) {
  if (use_LorentzAngle_DB_) {  // Get Lorentz angle from DB record
    siPhase2OTLorentzAngle_ = &es.getData(siPhase2OTLorentzAngleToken_);
  }

  geom_ = &es.getData(geomToken_);
}

PSPDigitizerAlgorithm::PSPDigitizerAlgorithm(const edm::ParameterSet& conf, edm::ConsumesCollector iC)
    : Phase2TrackerDigitizerAlgorithm(conf.getParameter<ParameterSet>("AlgorithmCommon"),
                                      conf.getParameter<ParameterSet>("PSPDigitizerAlgorithm"),
                                      iC),
      geomToken_(iC.esConsumes()),
      biasRailInefficiencyFlag_(
          conf.getParameter<ParameterSet>("PSPDigitizerAlgorithm").getParameter<int>("BiasRailInefficiencyFlag")) {
  if (use_LorentzAngle_DB_)
    siPhase2OTLorentzAngleToken_ = iC.esConsumes();
  pixelFlag_ = false;
  LogDebug("PSPDigitizerAlgorithm") << "Algorithm constructed "
                                    << "Configuration parameters:"
                                    << "Threshold/Gain = "
                                    << "threshold in electron Endcap = " << theThresholdInE_Endcap_
                                    << "threshold in electron Barrel = " << theThresholdInE_Barrel_ << " "
                                    << theElectronPerADC_ << " " << theAdcFullScale_ << " The delta cut-off is set to "
                                    << tMax_ << " pix-inefficiency " << addPixelInefficiency_
                                    << "Bias Rail Inefficiency " << biasRailInefficiencyFlag_;
}
PSPDigitizerAlgorithm::~PSPDigitizerAlgorithm() { LogDebug("PSPDigitizerAlgorithm") << "Algorithm deleted"; }
//
// -- Select the Hit for Digitization (sigScale will be implemented in future)
//
bool PSPDigitizerAlgorithm::select_hit(const PSimHit& hit, double tCorr, double& sigScale) const {
  if (biasRailInefficiencyFlag_ > 0 && isInBiasRailRegion(hit))
    return false;
  double toa = hit.tof() - tCorr;
  return (toa > theTofLowerCut_ && toa < theTofUpperCut_);
}
//
// -- Compare Signal with Threshold
//
bool PSPDigitizerAlgorithm::isAboveThreshold(const digitizerUtility::SimHitInfo* hitInfo,
                                             float charge,
                                             float thr) const {
  return (charge >= thr);
}
//
//  Check whether the Hit is in the Inefficient Bias Rail Region
//
bool PSPDigitizerAlgorithm::isInBiasRailRegion(const PSimHit& hit) const {
  constexpr float implant = 0.1467;  // Implant length (1.467 mm)
  constexpr float bRail = 0.00375;   // Bias Rail region which causes inefficiency (37.5micron)
  // Do coordinate transformation of the local Y from module middle point considering 32 implants and 31 inter-impant regions with bias rail
  constexpr float block_len = 16 * implant + 15.5 * bRail;
  constexpr float block_unit = implant + bRail;
  float yin = hit.entryPoint().y() + block_len;
  float yout = hit.exitPoint().y() + block_len;
  bool result = false;

  // Flag= 1 corresponds to optimistic case when the entire trajectory is withon the bias rail region the SimHit will be rejected
  if (biasRailInefficiencyFlag_ == 1 && (std::fmod(yin, block_unit) > implant && std::fmod(yout, block_unit) > implant))
    result = true;
  // Flag= 2 corresponds to pessimistic case i.e iven in a small part of the trajectory is inside the bias rain region the SimHit is rejected
  else if (biasRailInefficiencyFlag_ == 2 &&
           (std::fmod(yin, block_unit) > implant || std::fmod(yout, block_unit) > implant))
    result = true;
  return result;
}
//
// -- Read Bad Channels from the Condidion DB and kill channels/module accordingly
//
void PSPDigitizerAlgorithm::module_killing_DB(const Phase2TrackerGeomDetUnit* pixdet) {
  // this method is dummy at the moment. Will be implemented once we have the corresponding objectcondition DB
}
