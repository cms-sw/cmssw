#include <iostream>
#include <cmath>

#include "SimTracker/SiPhase2Digitizer/plugins/PSSDigitizerAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerLorentzAngleRcd.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

using namespace edm;

void PSSDigitizerAlgorithm::init(const edm::EventSetup& es) {
  if (use_LorentzAngle_DB_)  // Get Lorentz angle from DB record
    es.get<SiPhase2OuterTrackerLorentzAngleSimRcd>().get(SiPhase2OTLorentzAngle_);

  es.get<TrackerDigiGeometryRecord>().get(geom_);
}
PSSDigitizerAlgorithm::PSSDigitizerAlgorithm(const edm::ParameterSet& conf)
    : Phase2TrackerDigitizerAlgorithm(conf.getParameter<ParameterSet>("AlgorithmCommon"),
                                      conf.getParameter<ParameterSet>("PSSDigitizerAlgorithm")) {
  pixelFlag_ = false;
  LogDebug("PSSDigitizerAlgorithm") << "Algorithm constructed "
                                    << "Configuration parameters: "
                                    << "Threshold/Gain = "
                                    << "threshold in electron Endcap = " << theThresholdInE_Endcap_
                                    << "threshold in electron Barrel = " << theThresholdInE_Barrel_ << " "
                                    << theElectronPerADC_ << " " << theAdcFullScale_ << " The delta cut-off is set to "
                                    << tMax_ << " pix-inefficiency " << addPixelInefficiency_;
}
PSSDigitizerAlgorithm::~PSSDigitizerAlgorithm() { LogDebug("PSSDigitizerAlgorithm") << "Algorithm deleted"; }
//
// -- Select the Hit for Digitization (sigScale will be implemented in future)
//
bool PSSDigitizerAlgorithm::select_hit(const PSimHit& hit, double tCorr, double& sigScale) const {
  double toa = hit.tof() - tCorr;
  return (toa > theTofLowerCut_ && toa < theTofUpperCut_);
}
//
// -- Compare Signal with Threshold
//
bool PSSDigitizerAlgorithm::isAboveThreshold(const DigitizerUtility::SimHitInfo* const hisInfo,
                                             float charge,
                                             float thr) const {
  return (charge >= thr);
}
