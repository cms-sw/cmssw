#ifndef _SimTracker_SiPhase2Digitizer_PSSDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_PSSDigitizerAlgorithm_h

#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"

class PSSDigitizerAlgorithm : public Phase2TrackerDigitizerAlgorithm {
public:
  PSSDigitizerAlgorithm(const edm::ParameterSet& conf);
  ~PSSDigitizerAlgorithm() override;

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es) override;

  bool select_hit(const PSimHit& hit, double tCorr, double& sigScale) const override;
  bool isAboveThreshold(const DigitizerUtility::SimHitInfo* hitInfo, float charge, float thr) const override;
};
#endif
