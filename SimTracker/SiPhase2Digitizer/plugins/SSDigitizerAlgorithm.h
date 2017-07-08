#ifndef _SimTracker_SiPhase2Digitizer_SSDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_SSDigitizerAlgorithm_h

#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerAlgorithm.h"


// forward declarations
class TrackerTopology;

class SSDigitizerAlgorithm :public Phase2TrackerDigitizerAlgorithm {
 public:
  SSDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine&);
  virtual ~SSDigitizerAlgorithm();

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es);
  
  //run the algorithm to digitize a single det
  void accumulateSimHits(std::vector<PSimHit>::const_iterator inputBegin,
                         std::vector<PSimHit>::const_iterator inputEnd,
                         size_t inputBeginGlobalIndex,
			 unsigned int tofBin,
                         const Phase2TrackerGeomDetUnit* pixdet,
                         const GlobalVector& bfield);
};
#endif
