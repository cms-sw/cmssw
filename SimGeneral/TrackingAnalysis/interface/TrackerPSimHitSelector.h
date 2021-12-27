#ifndef TrackingAnalysis_TrackerPSimHitSelector_h
#define TrackingAnalysis_TrackerPSimHitSelector_h

#include "SimGeneral/TrackingAnalysis/interface/PSimHitSelector.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

//! TrackerPSimHitSelector class
class TrackerPSimHitSelector : public PSimHitSelector {
public:
  //! Constructor by pset.
  /* Creates a TrackerPSimHitSelector with association given by pset.

     /param[in] pset with the configuration values
  */
  TrackerPSimHitSelector(edm::ParameterSet const &config, edm::ConsumesCollector &iC)
      : PSimHitSelector(config, iC), cableToken_(iC.esConsumes()) {}

  //! Pre-process event information
  void select(PSimHitCollection &, edm::Event const &, edm::EventSetup const &) const override;

private:
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> cableToken_;
};

#endif
