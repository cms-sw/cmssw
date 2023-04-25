#ifndef SimTracker_SiPhase2Digitizer_PSSDigitizerAlgorithm_h
#define SimTracker_SiPhase2Digitizer_PSSDigitizerAlgorithm_h

#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerLorentzAngleRcd.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

class PSSDigitizerAlgorithm : public Phase2TrackerDigitizerAlgorithm {
public:
  PSSDigitizerAlgorithm(const edm::ParameterSet& conf, edm::ConsumesCollector iC);
  ~PSSDigitizerAlgorithm() override;

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es) override;

  bool select_hit(const PSimHit& hit, double tCorr, double& sigScale) const override;
  bool isAboveThreshold(const digitizerUtility::SimHitInfo* hitInfo, float charge, float thr) const override;
  void module_killing_DB(const Phase2TrackerGeomDetUnit* pixdet) override;

private:
  edm::ESGetToken<SiPhase2OuterTrackerLorentzAngle, SiPhase2OuterTrackerLorentzAngleSimRcd> siPhase2OTLorentzAngleToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<SiStripBadStrip, SiPhase2OuterTrackerBadStripRcd> badChannelToken_;
  const SiStripBadStrip* badChannelPayload_;
};
#endif
