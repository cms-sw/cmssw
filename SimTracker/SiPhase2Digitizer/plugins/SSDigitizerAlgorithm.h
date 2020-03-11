#ifndef _SimTracker_SiPhase2Digitizer_SSDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_SSDigitizerAlgorithm_h

#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"

class SSDigitizerAlgorithm : public Phase2TrackerDigitizerAlgorithm {
public:
  SSDigitizerAlgorithm(const edm::ParameterSet& conf);
  ~SSDigitizerAlgorithm() override;

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es) override;

  // run the algorithm to digitize a single det
  void accumulateSimHits(const std::vector<PSimHit>::const_iterator inputBegin,
                         const std::vector<PSimHit>::const_iterator inputEnd,
                         const size_t inputBeginGlobalIndex,
                         const uint32_t tofBin,
                         const Phase2TrackerGeomDetUnit* pixdet,
                         const GlobalVector& bfield) override;
};
#endif
