#include <iostream>
#include <cmath>

#include "SimTracker/SiPhase2Digitizer/plugins/PSPDigitizerAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

using namespace edm;

void PSPDigitizerAlgorithm::init(const edm::EventSetup& es) { es.get<TrackerDigiGeometryRecord>().get(geom_); }
PSPDigitizerAlgorithm::PSPDigitizerAlgorithm(const edm::ParameterSet& conf)
    : Phase2TrackerDigitizerAlgorithm(conf.getParameter<ParameterSet>("AlgorithmCommon"),
                                      conf.getParameter<ParameterSet>("PSPDigitizerAlgorithm")) {
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
bool PSPDigitizerAlgorithm::isAboveThreshold(const DigitizerUtility::SimHitInfo* hitInfo, float charge, float thr) const {
  return (charge >= thr);
}
