#ifndef _SimTracker_SiPhase2Digitizer_PixelDigitizerAlgorithm_h
#define _SimTracker_SiPhase2Digitizer_PixelDigitizerAlgorithm_h

#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerDigitizerAlgorithm.h"
#include "SimTracker/SiPhase2Digitizer/interface/DigitizerUtility.h"
#include "SimTracker/SiPhase2Digitizer/interface/Phase2TrackerDigitizerFwd.h"

// forward declarations
// For the random numbers
//namespace CLHEP {
//  class HepRandomEngine;
//}

//namespace edm {
//  class EventSetup;
//  class ParameterSet;
//}

//class PixelDigi;
//class PixelDigiSimLink;
//class PixelGeomDetUnit;
//class TrackerTopology;

class PixelDigitizerAlgorithm: public Phase2TrackerDigitizerAlgorithm {
 private:
  // tpyedef to change names of current pixel types to phase 2 types
  typedef PixelGeomDetUnit Phase2TrackerGeomDetUnit;
  typedef PixelDigi Phase2TrackerDigi;
  typedef PixelDigiSimLink Phase2TrackerDigiSimLink;
  typedef PixelTopology Phase2TrackerTopology;

 public:
  PixelDigitizerAlgorithm(const edm::ParameterSet& conf, CLHEP::HepRandomEngine&);
  ~PixelDigitizerAlgorithm();

  // initialization that cannot be done in the constructor
  void init(const edm::EventSetup& es);
  
  // void initializeEvent();
  // run the algorithm to digitize a single det
  void accumulateSimHits(const std::vector<PSimHit>::const_iterator inputBegin,
                         const std::vector<PSimHit>::const_iterator inputEnd,
                         const Phase2TrackerGeomDetUnit* pixdet,
                         const GlobalVector& bfield);
  void digitize(const Phase2TrackerGeomDetUnit* pixdet,
                std::vector<Phase2TrackerDigi>& digis,
                std::vector<Phase2TrackerDigiSimLink>& simlinks,
		const TrackerTopology* tTopo);
};
#endif
