#include <iostream>
#include <cmath>

#include "SimTracker/SiPhase2Digitizer/plugins/PixelDigitizerAlgorithm.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineSimService.h"

// Geometry
#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"

using namespace edm;
using namespace sipixelobjects;

void PixelDigitizerAlgorithm::init(const edm::EventSetup& es) {
  if (use_ineff_from_db_)  // load gain calibration service fromdb...
    theSiPixelGainCalibrationService_->setESObjects(es);

  if (use_deadmodule_DB_)
    siPixelBadModule_ = &es.getData(siPixelBadModuleToken_);

  if (use_LorentzAngle_DB_)  // Get Lorentz angle from DB record
    siPixelLorentzAngle_ = &es.getData(siPixelLorentzAngleToken_);

  // gets the map and geometry from the DB (to kill ROCs)
  fedCablingMap_ = &es.getData(fedCablingMapToken_);
  geom_ = &es.getData(geomToken_);
  if (useChargeReweighting_) {
    theSiPixelChargeReweightingAlgorithm_->init(es);
  }
}

PixelDigitizerAlgorithm::PixelDigitizerAlgorithm(const edm::ParameterSet& conf, edm::ConsumesCollector iC)
    : Phase2TrackerDigitizerAlgorithm(conf.getParameter<ParameterSet>("AlgorithmCommon"),
                                      conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm"),
                                      iC),
      odd_row_interchannelCoupling_next_row_(conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm")
                                                 .getParameter<double>("Odd_row_interchannelCoupling_next_row")),
      even_row_interchannelCoupling_next_row_(conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm")
                                                  .getParameter<double>("Even_row_interchannelCoupling_next_row")),
      odd_column_interchannelCoupling_next_column_(
          conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm")
              .getParameter<double>("Odd_column_interchannelCoupling_next_column")),
      even_column_interchannelCoupling_next_column_(
          conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm")
              .getParameter<double>("Even_column_interchannelCoupling_next_column")),
      apply_timewalk_(conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm").getParameter<bool>("ApplyTimewalk")),
      timewalk_model_(
          conf.getParameter<ParameterSet>("PixelDigitizerAlgorithm").getParameter<edm::ParameterSet>("TimewalkModel")),
      fedCablingMapToken_(iC.esConsumes()),
      geomToken_(iC.esConsumes()) {
  if (use_deadmodule_DB_)
    siPixelBadModuleToken_ = iC.esConsumes();
  if (use_LorentzAngle_DB_)
    siPixelLorentzAngleToken_ = iC.esConsumes();
  pixelFlag_ = true;
  LogDebug("PixelDigitizerAlgorithm") << "Algorithm constructed "
                                      << "Configuration parameters:"
                                      << "Threshold/Gain = "
                                      << "threshold in electron Endcap = " << theThresholdInE_Endcap_
                                      << "threshold in electron Barrel = " << theThresholdInE_Barrel_ << " "
                                      << theElectronPerADC_ << " " << theAdcFullScale_
                                      << " The delta cut-off is set to " << tMax_ << " pix-inefficiency "
                                      << addPixelInefficiency_;
}

PixelDigitizerAlgorithm::~PixelDigitizerAlgorithm() { LogDebug("PixelDigitizerAlgorithm") << "Algorithm deleted"; }

//
// -- Select the Hit for Digitization
//
bool PixelDigitizerAlgorithm::select_hit(const PSimHit& hit, double tCorr, double& sigScale) const {
  // in case of signal-shape emulation do not apply [TofLower,TofUpper] selection 
  double toa = hit.tof() - tCorr;
  return apply_timewalk_ || (toa >= theTofLowerCut_ && toa < theTofUpperCut_);

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
                          digitizerUtility::Ph2Amplitude(signalInElectrons_even_row_Xtalk_next_row, nullptr, -1.0));
      else
        signalNew.emplace(chanXtalkPrev,
                          digitizerUtility::Ph2Amplitude(signalInElectrons_odd_row_Xtalk_next_row, nullptr, -1.0));
    }
    if (hitChan.first < numRows - 1) {
      auto XtalkNext = std::make_pair(hitChan.first + 1, hitChan.second);
      int chanXtalkNext = pixelFlag_ ? PixelDigi::pixelToChannel(XtalkNext.first, XtalkNext.second)
                                     : Phase2TrackerDigi::pixelToChannel(XtalkNext.first, XtalkNext.second);
      if (hitChan.first % 2 == 1)
        signalNew.emplace(chanXtalkNext,
                          digitizerUtility::Ph2Amplitude(signalInElectrons_odd_row_Xtalk_next_row, nullptr, -1.0));
      else
        signalNew.emplace(chanXtalkNext,
                          digitizerUtility::Ph2Amplitude(signalInElectrons_even_row_Xtalk_next_row, nullptr, -1.0));
    }

    if (hitChan.second != 0) {
      auto XtalkPrev = std::make_pair(hitChan.first, hitChan.second - 1);
      int chanXtalkPrev = pixelFlag_ ? PixelDigi::pixelToChannel(XtalkPrev.first, XtalkPrev.second)
                                     : Phase2TrackerDigi::pixelToChannel(XtalkPrev.first, XtalkPrev.second);
      if (hitChan.second % 2 == 1)
        signalNew.emplace(
            chanXtalkPrev,
            digitizerUtility::Ph2Amplitude(signalInElectrons_even_column_Xtalk_next_column, nullptr, -1.0));
      else
        signalNew.emplace(
            chanXtalkPrev,
            digitizerUtility::Ph2Amplitude(signalInElectrons_odd_column_Xtalk_next_column, nullptr, -1.0));
    }
    if (hitChan.second < numColumns - 1) {
      auto XtalkNext = std::make_pair(hitChan.first, hitChan.second + 1);
      int chanXtalkNext = pixelFlag_ ? PixelDigi::pixelToChannel(XtalkNext.first, XtalkNext.second)
                                     : Phase2TrackerDigi::pixelToChannel(XtalkNext.first, XtalkNext.second);
      if (hitChan.second % 2 == 1)
        signalNew.emplace(
            chanXtalkNext,
            digitizerUtility::Ph2Amplitude(signalInElectrons_odd_column_Xtalk_next_column, nullptr, -1.0));
      else
        signalNew.emplace(
            chanXtalkNext,
            digitizerUtility::Ph2Amplitude(signalInElectrons_even_column_Xtalk_next_column, nullptr, -1.0));
    }
  }
  for (auto const& l : signalNew) {
    int chan = l.first;
    auto iter = theSignal.find(chan);
    if (iter != theSignal.end()) {
      iter->second += l.second.ampl();
    } else {
      theSignal.emplace(chan, digitizerUtility::Ph2Amplitude(l.second.ampl(), nullptr, -1.0));
    }
  }
}

PixelDigitizerAlgorithm::TimewalkCurve::TimewalkCurve(const edm::ParameterSet& pset)
    : x_(pset.getParameter<std::vector<double>>("charge")), y_(pset.getParameter<std::vector<double>>("delay")) {
  if (x_.size() != y_.size())
    throw cms::Exception("Configuration")
        << "Timewalk model error: the number of charge values does not match the number of delay values!";
}

double PixelDigitizerAlgorithm::TimewalkCurve::operator()(double x) const {
  auto it = std::lower_bound(x_.begin(), x_.end(), x);
  if (it == x_.begin())
    return y_.front();
  if (it == x_.end())
    return y_.back();
  int index = std::distance(x_.begin(), it);
  double x_high = *it;
  double x_low = *(--it);
  double p = (x - x_low) / (x_high - x_low);
  return p * y_[index] + (1 - p) * y_[index - 1];
}

PixelDigitizerAlgorithm::TimewalkModel::TimewalkModel(const edm::ParameterSet& pset) {
  threshold_values = pset.getParameter<std::vector<double>>("ThresholdValues");
  const auto& curve_psetvec = pset.getParameter<std::vector<edm::ParameterSet>>("Curves");
  if (threshold_values.size() != curve_psetvec.size())
    throw cms::Exception("Configuration")
        << "Timewalk model error: the number of threshold values does not match the number of curves.";
  for (const auto& curve_pset : curve_psetvec)
    curves.emplace_back(curve_pset);
}

double PixelDigitizerAlgorithm::TimewalkModel::operator()(double q_in, double q_threshold) const {
  auto index = find_closest_index(threshold_values, q_threshold);
  return curves[index](q_in);
}

std::size_t PixelDigitizerAlgorithm::TimewalkModel::find_closest_index(const std::vector<double>& vec,
                                                                       double value) const {
  auto it = std::lower_bound(vec.begin(), vec.end(), value);
  if (it == vec.begin())
    return 0;
  else if (it == vec.end())
    return vec.size() - 1;
  else {
    auto it_upper = it;
    auto it_lower = --it;
    auto closest = (value - *it_lower > *it_upper - value) ? it_upper : it_lower;
    return std::distance(vec.begin(), closest);
  }
}
//
// -- Compare Signal with Threshold
//
bool PixelDigitizerAlgorithm::isAboveThreshold(const digitizerUtility::SimHitInfo* hitInfo,
                                               float charge,
                                               float thr) const {
  if (charge < thr)
    return false;
  if (apply_timewalk_ && hitInfo) {
    float corrected_time = hitInfo->time();
    double time = corrected_time + timewalk_model_(charge, thr);
    return (time >= theTofLowerCut_ && time < theTofUpperCut_);
  } else
    return true;
}
//
// -- Read Bad Channels from the Condidion DB and kill channels/module accordingly
//
void PixelDigitizerAlgorithm::module_killing_DB(const Phase2TrackerGeomDetUnit* pixdet) {
  bool isbad = false;
  uint32_t detID = pixdet->geographicalId().rawId();
  int ncol = pixdet->specificTopology().ncolumns();
  if (ncol < 0)
    return;
  std::vector<SiPixelQuality::disabledModuleType> disabledModules = siPixelBadModule_->getBadComponentList();

  SiPixelQuality::disabledModuleType badmodule;
  for (const auto& mod : disabledModules) {
    if (detID == mod.DetID) {
      isbad = true;
      badmodule = mod;
      break;
    }
  }

  if (!isbad)
    return;

  signal_map_type& theSignal = _signal[detID];  // check validity
  if (badmodule.errorType == 0) {               // this is a whole dead module.
    for (auto& s : theSignal)
      s.second.set(0.);  // reset amplitude
  } else {               // all other module types: half-modules and single ROCs.
    // Get Bad ROC position:
    // follow the example of getBadRocPositions in CondFormats/SiPixelObjects/src/SiPixelQuality.cc
    std::vector<GlobalPixel> badrocpositions;
    for (size_t j = 0; j < static_cast<size_t>(ncol); j++) {
      if (siPixelBadModule_->IsRocBad(detID, j)) {
        std::vector<CablingPathToDetUnit> path = fedCablingMap_->pathToDetUnit(detID);
        for (auto const& p : path) {
          const PixelROC* myroc = fedCablingMap_->findItem(p);
          if (myroc->idInDetUnit() == j) {
            LocalPixel::RocRowCol local = {39, 25};  //corresponding to center of ROC row, col
            GlobalPixel global = myroc->toGlobal(LocalPixel(local));
            badrocpositions.push_back(global);
            break;
          }
        }
      }
    }

    for (auto& s : theSignal) {
      std::pair<int, int> ip;
      if (pixelFlag_)
        ip = PixelDigi::channelToPixel(s.first);
      else
        ip = Phase2TrackerDigi::channelToPixel(s.first);

      for (auto const& p : badrocpositions) {
        for (auto& k : badPixels_) {
          if (p.row == k.getParameter<int>("row") && ip.first == k.getParameter<int>("row") &&
              std::abs(ip.second - p.col) < k.getParameter<int>("col")) {
            s.second.set(0.);
          }
        }
      }
    }
  }
}
