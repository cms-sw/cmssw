#include <iostream>
#include <cmath>

#include "SimTracker/SiPhase2Digitizer/plugins/SSDigitizerAlgorithm.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Geometry
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerCondDataRecords.h"

using namespace edm;

namespace {
  double nFactorial(int n);
  double aScalingConstant(int N, int i);
}  // namespace
namespace {
  double nFactorial(int n) { return std::tgamma(n + 1); }
  double aScalingConstant(int N, int i) {
    return std::pow(-1, (double)i) * nFactorial(N) * nFactorial(N + 2) /
           (nFactorial(N - i) * nFactorial(N + 2 - i) * nFactorial(i));
  }
}  // namespace

void SSDigitizerAlgorithm::init(const edm::EventSetup& es) {
  if (use_LorentzAngle_DB_)  // Get Lorentz angle from DB record
    siPhase2OTLorentzAngle_ = &es.getData(siPhase2OTLorentzAngleToken_);

  if (use_deadmodule_DB_)  // Get Bad Channel (SiStripBadStrip) from DB
    badChannelPayload_ = &es.getData(badChannelToken_);

  geom_ = &es.getData(geomToken_);
}

SSDigitizerAlgorithm::SSDigitizerAlgorithm(const edm::ParameterSet& conf, edm::ConsumesCollector iC)
    : Phase2TrackerDigitizerAlgorithm(conf.getParameter<ParameterSet>("AlgorithmCommon"),
                                      conf.getParameter<ParameterSet>("SSDigitizerAlgorithm"),
                                      iC),
      hitDetectionMode_(conf.getParameter<ParameterSet>("SSDigitizerAlgorithm").getParameter<int>("HitDetectionMode")),
      pulseShapeParameters_(conf.getParameter<ParameterSet>("SSDigitizerAlgorithm")
                                .getParameter<std::vector<double> >("PulseShapeParameters")),
      deadTime_(conf.getParameter<ParameterSet>("SSDigitizerAlgorithm").getParameter<double>("CBCDeadTime")),
      geomToken_(iC.esConsumes()) {
  if (use_LorentzAngle_DB_)
    siPhase2OTLorentzAngleToken_ = iC.esConsumes();

  if (use_deadmodule_DB_) {
    std::string badChannelLabel_ = conf.getParameter<ParameterSet>("SSDigitizerAlgorithm")
                                       .getUntrackedParameter<std::string>("BadChannelLabel", "");
    badChannelToken_ = iC.esConsumes(edm::ESInputTag{"", badChannelLabel_});
  }

  pixelFlag_ = false;
  LogDebug("SSDigitizerAlgorithm ") << "SSDigitizerAlgorithm constructed "
                                    << "Configuration parameters:"
                                    << "Threshold/Gain = "
                                    << "threshold in electron Endcap = " << theThresholdInE_Endcap_
                                    << "threshold in electron Barrel = " << theThresholdInE_Barrel_ << " "
                                    << theElectronPerADC_ << " " << theAdcFullScale_ << " The delta cut-off is set to "
                                    << tMax_ << " pix-inefficiency " << addPixelInefficiency_;
  storeSignalShape();
}
SSDigitizerAlgorithm::~SSDigitizerAlgorithm() { LogDebug("SSDigitizerAlgorithm") << "SSDigitizerAlgorithm deleted"; }
//
// -- Select the Hit for Digitization
//
bool SSDigitizerAlgorithm::select_hit(const PSimHit& hit, double tCorr, double& sigScale) const {
  bool result = false;
  if (hitDetectionMode_ == SSDigitizerAlgorithm::SampledMode)
    result = select_hit_sampledMode(hit, tCorr, sigScale);
  else if (hitDetectionMode_ == SSDigitizerAlgorithm::LatchedMode)
    result = select_hit_latchedMode(hit, tCorr, sigScale);
  else {
    double toa = hit.tof() - tCorr;
    result = (toa > theTofLowerCut_ && toa < theTofUpperCut_);
  }
  return result;
}
//
// -- Select Hits in Sampled Mode
//
bool SSDigitizerAlgorithm::select_hit_sampledMode(const PSimHit& hit, double tCorr, double& sigScale) const {
  double toa = hit.tof() - tCorr;
  double sampling_time = bx_time;

  DetId det_id = DetId(hit.detUnitId());
  float theThresholdInE =
      (det_id.subdetId() == StripSubdetector::TOB) ? theThresholdInE_Barrel_ : theThresholdInE_Endcap_;

  sigScale = getSignalScale(sampling_time - toa);
  return (sigScale * hit.energyLoss() / GeVperElectron_ > theThresholdInE);
}
//
// -- Select Hits in Hit Detection Mode
//
bool SSDigitizerAlgorithm::select_hit_latchedMode(const PSimHit& hit, double tCorr, double& sigScale) const {
  float toa = hit.tof() - tCorr;
  toa -= hit.eventId().bunchCrossing() * bx_time;

  float sampling_time = (-1) * (hit.eventId().bunchCrossing() + 1) * bx_time;

  DetId det_id = DetId(hit.detUnitId());
  float theThresholdInE =
      (det_id.subdetId() == StripSubdetector::TOB) ? theThresholdInE_Barrel_ : theThresholdInE_Endcap_;

  bool lastPulse = true;
  bool aboveThr = false;
  for (float i = deadTime_; i <= bx_time; i++) {
    sigScale = getSignalScale(sampling_time - toa + i);

    aboveThr = (sigScale * hit.energyLoss() / GeVperElectron_ > theThresholdInE);
    if (!lastPulse && aboveThr)
      return true;

    lastPulse = aboveThr;
  }
  return false;
}
double SSDigitizerAlgorithm::cbc3PulsePolarExpansion(double x) const {
  constexpr size_t max_par = 6;
  if (pulseShapeParameters_.size() < max_par)
    return -1;
  double xOffset = pulseShapeParameters_[0];
  double tau = pulseShapeParameters_[1];
  double r = pulseShapeParameters_[2];
  double theta = pulseShapeParameters_[3];
  int nTerms = static_cast<int>(pulseShapeParameters_[4]);

  double fN = 0;
  double xx = x - xOffset;
  if (xx < 0)
    return 0;

  for (int i = 0; i < nTerms; i++) {
    double angularTerm = 0;
    double temporalTerm = 0;
    double rTerm = std::pow(r, i) / (std::pow(tau, 2. * i) * ::nFactorial(i + 2));
    for (int j = 0; j <= i; j++) {
      angularTerm += std::pow(std::cos(theta), (double)(i - j)) * std::pow(std::sin(theta), (double)j);
      temporalTerm += ::aScalingConstant(i, j) * std::pow(xx, (double)(i - j)) * std::pow(tau, (double)j);
    }
    double fi = rTerm * angularTerm * temporalTerm;

    fN += fi;
  }
  return fN;
}
double SSDigitizerAlgorithm::signalShape(double x) const {
  double xOffset = pulseShapeParameters_[0];
  double tau = pulseShapeParameters_[1];
  double maxCharge = pulseShapeParameters_[5];

  double xx = x - xOffset;
  return maxCharge * (std::exp(-xx / tau) * std::pow(xx / tau, 2.) * cbc3PulsePolarExpansion(x));
}
void SSDigitizerAlgorithm::storeSignalShape() {
  for (size_t i = 0; i < interpolationPoints; i++) {
    float val = i / interpolationStep;

    pulseShapeVec_.push_back(signalShape(val));
  }
}
double SSDigitizerAlgorithm::getSignalScale(double xval) const {
  double res = 0.0;
  int len = pulseShapeVec_.size();

  if (xval < 0.0 || xval * interpolationStep >= len)
    return res;

  unsigned int lower = std::floor(xval) * interpolationStep;
  unsigned int upper = std::ceil(xval) * interpolationStep;
  for (size_t i = lower + 1; i < upper * interpolationStep; i++) {
    float val = i * 0.1;
    if (val > xval) {
      res = pulseShapeVec_[i - 1];
      break;
    }
  }
  return res;
}
//
// -- Compare Signal with Threshold
//
bool SSDigitizerAlgorithm::isAboveThreshold(const digitizerUtility::SimHitInfo* const hisInfo,
                                            float charge,
                                            float thr) const {
  return (charge >= thr);
}
//
// -- Read Bad Channels from the Condidion DB and kill channels/module accordingly
//
void SSDigitizerAlgorithm::module_killing_DB(const Phase2TrackerGeomDetUnit* pixdet) {
  uint32_t detId = pixdet->geographicalId().rawId();

  signal_map_type& theSignal = _signal[detId];
  signal_map_type signalNew;

  SiStripBadStrip::Range range = badChannelPayload_->getRange(detId);
  for (std::vector<unsigned int>::const_iterator badChannel = range.first; badChannel != range.second; ++badChannel) {
    const auto& firstStrip = badChannelPayload_->decodePhase2(*badChannel).firstStrip;
    const auto& channelRange = badChannelPayload_->decodePhase2(*badChannel).range;

    for (int index = 0; index < channelRange; index++) {
      for (auto& s : theSignal) {
        auto& channel = s.first;
        if (channel == firstStrip + index)
          s.second.set(0.);
      }
    }
  }
}
