#ifndef _SimTracker_SiPhase2Digitizer_PixelDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_PixelDigitizerAlgorithm_h

#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"

class PixelDigitizerAlgorithm : public Phase2TrackerDigitizerAlgorithm {
public:
  PixelDigitizerAlgorithm(const edm::ParameterSet& conf);
  ~PixelDigitizerAlgorithm() override;

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es) override;

  // void initializeEvent();
  // run the algorithm to digitize a single det
  void accumulateSimHits(const std::vector<PSimHit>::const_iterator inputBegin,
                         const std::vector<PSimHit>::const_iterator inputEnd,
                         const size_t inputBeginGlobalIndex,
                         const uint32_t tofBin,
                         const Phase2TrackerGeomDetUnit* pixdet,
                         const GlobalVector& bfield) override;
  void add_cross_talk(const Phase2TrackerGeomDetUnit* pixdet) override;

  // Addition four xtalk-related parameters to PixelDigitizerAlgorithm specific parameters initialized in Phase2TrackerDigitizerAlgorithm
  const double odd_row_interchannelCoupling_next_row_;
  const double even_row_interchannelCoupling_next_row_;
  const double odd_column_interchannelCoupling_next_column_;
  const double even_column_interchannelCoupling_next_column_;
};
#endif
