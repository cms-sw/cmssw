#ifndef _SimTracker_SiPhase2Digitizer_PSSDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_PSSDigitizerAlgorithm_h

#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerDigitizerAlgorithm.h"
#include "SimTracker/SiPhase2Digitizer/interface/DigitizerUtility.h"
#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerDigitizerFwd.h"

// forward declarations
// For the random numbers
//namespace CLHEP {
//  class HepRandomEngine;
//}
//
//namespace edm {
//  class EventSetup;
//  class ParameterSet;
//}
//
//class PixelDigi;
//class PixelDigiSimLink;
//class PixelGeomDetUnit;
//class TrackerTopology;

class PSSDigitizerAlgorithm :public Phase2TrackerDigitizerAlgorithm {
 public:
  PSSDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine&);
  ~PSSDigitizerAlgorithm();

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es);
  
  // void initializeEvent();
  // run the algorithm to digitize a single det
  void accumulateSimHits(const std::vector<PSimHit>::const_iterator inputBegin,
                         const std::vector<PSimHit>::const_iterator inputEnd,
                         const Phase2TrackerGeomDetUnit* pixdet,
                         const GlobalVector& bfield);
  void digitize(const PixelGeomDetUnit* pixdet,
                std::vector<Phase2TrackerDigi>& digis,
                std::vector<Phase2TrackerDigiSimLink>& simlinks,
		const TrackerTopology* tTopo);
};
#endif
