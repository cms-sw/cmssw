#ifndef TrackingAnalysis_MuonPSimHitSelector_h
#define TrackingAnalysis_MuonPSimHitSelector_h

#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "CondFormats/DataRecord/interface/CSCBadChambersRcd.h"

#include "SimGeneral/TrackingAnalysis/interface/PSimHitSelector.h"

//! MuonPSimHitSelector class
class MuonPSimHitSelector : public PSimHitSelector {
public:
  //! Constructor by pset.
  /* Creates a MuonPSimHitSelector with association given by pset.

     /param[in] pset with the configuration values
  */
  MuonPSimHitSelector(edm::ParameterSet const &config, edm::ConsumesCollector &iC)
      : PSimHitSelector(config, iC), cscBadToken_(iC.esConsumes()) {}

  //! Pre-process event information
  void select(PSimHitCollection &, edm::Event const &, edm::EventSetup const &) const override;

private:
  const edm::ESGetToken<CSCBadChambers, CSCBadChambersRcd> cscBadToken_;
};

#endif
