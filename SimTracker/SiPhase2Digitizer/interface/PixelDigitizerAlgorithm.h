#ifndef _SimTracker_SiPhase2Digitizer_PixelDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_PixelDigitizerAlgorithm_h

#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerDigitizerAlgorithm.h"
#include "SimTracker/SiPhase2Digitizer/interface/DigitizerUtility.h"
#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerDigitizerFwd.h"

// forward declarations
class TrackerTopology;

class PixelDigitizerAlgorithm: public Phase2TrackerDigitizerAlgorithm {
 public:
  PixelDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine&);
  ~PixelDigitizerAlgorithm();

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es);
  
  // void initializeEvent();
  // run the algorithm to digitize a single det
  void accumulateSimHits(const std::vector<PSimHit>::const_iterator inputBegin,
                         const std::vector<PSimHit>::const_iterator inputEnd,
                         const size_t inputBeginGlobalIndex,
			 const unsigned int tofBin,
                         const Phase2TrackerGeomDetUnit* pixdet,
                         const GlobalVector& bfield);
};
#endif
