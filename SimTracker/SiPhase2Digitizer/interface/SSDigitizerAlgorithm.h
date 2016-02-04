#ifndef _SimTracker_SiPhase2Digitizer_SSDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_SSDigitizerAlgorithm_h

#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerDigitizerAlgorithm.h"
#include "SimTracker/SiPhase2Digitizer/interface/DigitizerUtility.h"
#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerDigitizerFwd.h"

// forward declarations
class TrackerTopology;

class SSDigitizerAlgorithm :public Phase2TrackerDigitizerAlgorithm {
 public:
  SSDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine&);
  ~SSDigitizerAlgorithm();

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es);
  
  //run the algorithm to digitize a single det
  void accumulateSimHits(const std::vector<PSimHit>::const_iterator inputBegin,
                         const std::vector<PSimHit>::const_iterator inputEnd,
                         const size_t inputBeginGlobalIndex,
			 const unsigned int tofBin,
                         const Phase2TrackerGeomDetUnit* pixdet,
                         const GlobalVector& bfield);
};
#endif
