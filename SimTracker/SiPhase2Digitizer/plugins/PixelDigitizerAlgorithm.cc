#include <iostream>
#include <cmath>

#include "SimTracker/SiPhase2Digitizer/plugins/PixelDigitizerAlgorithm.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineSimService.h"

#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

using namespace edm;
using namespace sipixelobjects;

void PixelDigitizerAlgorithm::init(const edm::EventSetup& es) {
  if (use_ineff_from_db_)  // load gain calibration service fromdb...
    theSiPixelGainCalibrationService_->setESObjects(es);

  if (use_deadmodule_DB_)
    es.get<SiPixelQualityRcd>().get(SiPixelBadModule_);

  if (use_LorentzAngle_DB_)  // Get Lorentz angle from DB record
    es.get<SiPixelLorentzAngleSimRcd>().get(SiPixelLorentzAngle_);

  // gets the map and geometry from the DB (to kill ROCs)
  es.get<SiPixelFedCablingMapRcd>().get(fedCablingMap_);
  es.get<TrackerDigiGeometryRecord>().get(geom_);
}

PixelDigitizerAlgorithm::PixelDigitizerAlgorithm(const edm::ParameterSet& conf)
    : Phase2TrackerDigitizerAlgorithm(conf.getParameter<ParameterSet>("AlgorithmCommon"),
                                      conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm")),
      odd_row_interchannelCoupling_next_row_(conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm")
                                                 .getParameter<double>("Odd_row_interchannelCoupling_next_row")),
      even_row_interchannelCoupling_next_row_(conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm")
                                                  .getParameter<double>("Even_row_interchannelCoupling_next_row")),
      odd_column_interchannelCoupling_next_column_(
          conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm")
              .getParameter<double>("Odd_column_interchannelCoupling_next_column")),
      even_column_interchannelCoupling_next_column_(
          conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm")
              .getParameter<double>("Even_column_interchannelCoupling_next_column")) {
  pixelFlag_ = true;
  LogInfo("PixelDigitizerAlgorithm") << "Algorithm constructed "
                                     << "Configuration parameters:"
                                     << "Threshold/Gain = "
                                     << "threshold in electron Endcap = " << theThresholdInE_Endcap_
                                     << "threshold in electron Barrel = " << theThresholdInE_Barrel_ << " "
                                     << theElectronPerADC_ << " " << theAdcFullScale_ << " The delta cut-off is set to "
                                     << tMax_ << " pix-inefficiency " << addPixelInefficiency_;
}
PixelDigitizerAlgorithm::~PixelDigitizerAlgorithm() { LogDebug("PixelDigitizerAlgorithm") << "Algorithm deleted"; }
void PixelDigitizerAlgorithm::accumulateSimHits(std::vector<PSimHit>::const_iterator inputBegin,
                                                std::vector<PSimHit>::const_iterator inputEnd,
                                                const size_t inputBeginGlobalIndex,
                                                const uint32_t tofBin,
                                                const Phase2TrackerGeomDetUnit* pixdet,
                                                const GlobalVector& bfield) {
  // produce SignalPoint's for all SimHit's in detector
  // Loop over hits
  uint32_t detId = pixdet->geographicalId().rawId();
  size_t simHitGlobalIndex = inputBeginGlobalIndex;  // This needs to be stored to create the digi-sim link later

  // find the relevant hits
  std::vector<PSimHit> matchedSimHits;
  std::copy_if(inputBegin, inputEnd, std::back_inserter(matchedSimHits), [detId](auto const& hit) -> bool {
    return hit.detUnitId() == detId;
  });
  // loop over a much reduced set of SimHits
  for (auto const& hit : matchedSimHits) {
    LogDebug("PixelDigitizerAlgorithm") << hit.particleType() << " " << hit.pabs() << " " << hit.energyLoss() << " "
                                        << hit.tof() << " " << hit.trackId() << " " << hit.processType() << " "
                                        << hit.detUnitId() << hit.entryPoint() << " " << hit.exitPoint();

    std::vector<DigitizerUtility::EnergyDepositUnit> ionization_points;
    std::vector<DigitizerUtility::SignalPoint> collection_points;

    // fill collection_points for this SimHit, indpendent of topology
    // Check the TOF cut
    if ((hit.tof() - pixdet->surface().toGlobal((hit).localPosition()).mag() / 30.) >= theTofLowerCut_ &&
        (hit.tof() - pixdet->surface().toGlobal((hit).localPosition()).mag() / 30.) <= theTofUpperCut_) {
      primary_ionization(hit, ionization_points);  // fills ionization_points

      // transforms ionization_points -> collection_points
      drift(hit, pixdet, bfield, ionization_points, collection_points);

      // compute induced signal on readout elements and add to _signal
      // hit needed only for SimHit<-->Digi link
      induce_signal(hit, simHitGlobalIndex, tofBin, pixdet, collection_points);
    }
    ++simHitGlobalIndex;
  }
}
// ======================================================================
//
//  Add  Cross-talk contribution
//
// ======================================================================
void PixelDigitizerAlgorithm::add_cross_talk(const Phase2TrackerGeomDetUnit* pixdet) {
  if (!pixelFlag_)
    return;

  const Phase2TrackerTopology* topol = &pixdet->specificTopology();

  // cross-talk calculation valid for the case of 25x100 pixels
  const float pitch_first = 0.0025;
  const float pitch_second = 0.0100;

  // 0.5 um tolerance when comparing the pitch to accommodate the small changes in different TK geometrie (temporary fix)
  const double pitch_tolerance(0.0005);

  if (std::abs(topol->pitch().first - pitch_first) > pitch_tolerance ||
      std::abs(topol->pitch().second - pitch_second) > pitch_tolerance)
    return;

  uint32_t detID = pixdet->geographicalId().rawId();
  signal_map_type& theSignal = _signal[detID];
  signal_map_type signalNew;

  int numRows = topol->nrows();
  int numColumns = topol->ncolumns();

  for (auto& s : theSignal) {
    float signalInElectrons = s.second.ampl();  // signal in electrons

    auto hitChan = PixelDigi::channelToPixel(s.first);

    float signalInElectrons_odd_row_Xtalk_next_row = signalInElectrons * odd_row_interchannelCoupling_next_row_;
    float signalInElectrons_even_row_Xtalk_next_row = signalInElectrons * even_row_interchannelCoupling_next_row_;
    float signalInElectrons_odd_column_Xtalk_next_column =
        signalInElectrons * odd_column_interchannelCoupling_next_column_;
    float signalInElectrons_even_column_Xtalk_next_column =
        signalInElectrons * even_column_interchannelCoupling_next_column_;

    // subtract the charge which will be shared
    s.second.set(signalInElectrons - signalInElectrons_odd_row_Xtalk_next_row -
                 signalInElectrons_even_row_Xtalk_next_row - signalInElectrons_odd_column_Xtalk_next_column -
                 signalInElectrons_even_column_Xtalk_next_column);

    if (hitChan.first != 0) {
      auto XtalkPrev = std::make_pair(hitChan.first - 1, hitChan.second);
      int chanXtalkPrev = pixelFlag_ ? PixelDigi::pixelToChannel(XtalkPrev.first, XtalkPrev.second)
                                     : Phase2TrackerDigi::pixelToChannel(XtalkPrev.first, XtalkPrev.second);
      if (hitChan.first % 2 == 1)
        signalNew.emplace(chanXtalkPrev,
                          DigitizerUtility::Amplitude(signalInElectrons_even_row_Xtalk_next_row, nullptr, -1.0));
      else
        signalNew.emplace(chanXtalkPrev,
                          DigitizerUtility::Amplitude(signalInElectrons_odd_row_Xtalk_next_row, nullptr, -1.0));
    }
    if (hitChan.first < numRows - 1) {
      auto XtalkNext = std::make_pair(hitChan.first + 1, hitChan.second);
      int chanXtalkNext = pixelFlag_ ? PixelDigi::pixelToChannel(XtalkNext.first, XtalkNext.second)
                                     : Phase2TrackerDigi::pixelToChannel(XtalkNext.first, XtalkNext.second);
      if (hitChan.first % 2 == 1)
        signalNew.emplace(chanXtalkNext,
                          DigitizerUtility::Amplitude(signalInElectrons_odd_row_Xtalk_next_row, nullptr, -1.0));
      else
        signalNew.emplace(chanXtalkNext,
                          DigitizerUtility::Amplitude(signalInElectrons_even_row_Xtalk_next_row, nullptr, -1.0));
    }

    if (hitChan.second != 0) {
      auto XtalkPrev = std::make_pair(hitChan.first, hitChan.second - 1);
      int chanXtalkPrev = pixelFlag_ ? PixelDigi::pixelToChannel(XtalkPrev.first, XtalkPrev.second)
                                     : Phase2TrackerDigi::pixelToChannel(XtalkPrev.first, XtalkPrev.second);
      if (hitChan.second % 2 == 1)
        signalNew.emplace(chanXtalkPrev,
                          DigitizerUtility::Amplitude(signalInElectrons_even_column_Xtalk_next_column, nullptr, -1.0));
      else
        signalNew.emplace(chanXtalkPrev,
                          DigitizerUtility::Amplitude(signalInElectrons_odd_column_Xtalk_next_column, nullptr, -1.0));
    }
    if (hitChan.second < numColumns - 1) {
      auto XtalkNext = std::make_pair(hitChan.first, hitChan.second + 1);
      int chanXtalkNext = pixelFlag_ ? PixelDigi::pixelToChannel(XtalkNext.first, XtalkNext.second)
                                     : Phase2TrackerDigi::pixelToChannel(XtalkNext.first, XtalkNext.second);
      if (hitChan.second % 2 == 1)
        signalNew.emplace(chanXtalkNext,
                          DigitizerUtility::Amplitude(signalInElectrons_odd_column_Xtalk_next_column, nullptr, -1.0));
      else
        signalNew.emplace(chanXtalkNext,
                          DigitizerUtility::Amplitude(signalInElectrons_even_column_Xtalk_next_column, nullptr, -1.0));
    }
  }
  for (auto const& l : signalNew) {
    int chan = l.first;
    auto iter = theSignal.find(chan);
    if (iter != theSignal.end()) {
      iter->second += l.second.ampl();
    } else {
      theSignal.emplace(chan, DigitizerUtility::Amplitude(l.second.ampl(), nullptr, -1.0));
    }
  }
}
