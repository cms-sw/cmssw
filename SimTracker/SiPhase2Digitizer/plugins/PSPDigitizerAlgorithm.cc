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
      geomToken_(iC.esConsumes()) {
  if (use_LorentzAngle_DB_)
    siPhase2OTLorentzAngleToken_ = iC.esConsumes();
  pixelFlag_ = false;
  LogDebug("PSPDigitizerAlgorithm") << "Algorithm constructed "
                                    << "Configuration parameters:"
                                    << "Threshold/Gain = "
                                    << "threshold in electron Endcap = " << theThresholdInE_Endcap_
                                    << "threshold in electron Barrel = " << theThresholdInE_Barrel_ << " "
                                    << theElectronPerADC_ << " " << theAdcFullScale_ << " The delta cut-off is set to "
                                    << tMax_ << " pix-inefficiency " << addPixelInefficiency_;
}
PSPDigitizerAlgorithm::~PSPDigitizerAlgorithm() { LogDebug("PSPDigitizerAlgorithm") << "Algorithm deleted"; }
//
// -- Select the Hit for Digitization (sigScale will be implemented in future)
//
bool PSPDigitizerAlgorithm::select_hit(const PSimHit& hit, double tCorr, double& sigScale) const {
  double toa = hit.tof() - tCorr;
  return (toa > theTofLowerCut_ && toa < theTofUpperCut_);
}
//
// -- Compare Signal with Threshold
//
bool PSPDigitizerAlgorithm::isAboveThreshold(const DigitizerUtility::SimHitInfo* hitInfo,
                                             float charge,
                                             float thr) const {
  return (charge >= thr);
}
