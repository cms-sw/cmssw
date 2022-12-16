#ifndef _SimTracker_SiPhase2Digitizer_SSDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_SSDigitizerAlgorithm_h

#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerLorentzAngleRcd.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

class SSDigitizerAlgorithm : public Phase2TrackerDigitizerAlgorithm {
public:
  SSDigitizerAlgorithm(const edm::ParameterSet& conf, edm::ConsumesCollector iC);
  ~SSDigitizerAlgorithm() override;

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es) override;

  bool select_hit(const PSimHit& hit, double tCorr, double& sigScale) const override;
  bool isAboveThreshold(const digitizerUtility::SimHitInfo* hitInfo, float charge, float thr) const override;

private:
  enum { SquareWindow, SampledMode, LatchedMode, SampledOrLachedMode, HIPFindingMode };
  double cbc3PulsePolarExpansion(double x) const;
  double signalShape(double x) const;
  double getSignalScale(double xval) const;
  void storeSignalShape();
  bool select_hit_sampledMode(const PSimHit& hit, double tCorr, double& sigScale) const;
  bool select_hit_latchedMode(const PSimHit& hit, double tCorr, double& sigScale) const;
  void module_killing_DB(const Phase2TrackerGeomDetUnit* pixdet) override;

  int hitDetectionMode_;
  std::vector<double> pulseShapeVec_;
  std::vector<double> pulseShapeParameters_;
  float deadTime_;
  edm::ESGetToken<SiPhase2OuterTrackerLorentzAngle, SiPhase2OuterTrackerLorentzAngleSimRcd> siPhase2OTLorentzAngleToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  static constexpr float bx_time{25};
  static constexpr size_t interpolationPoints{1000};
  static constexpr int interpolationStep{10};
  edm::ESGetToken<SiStripBadStrip, SiPhase2OuterTrackerBadStripRcd> badChannelToken_;
  const SiStripBadStrip* badChannelPayload_;
};
#endif
