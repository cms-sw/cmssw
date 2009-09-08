#ifndef TrackingAnalysis_MuonPSimHitSelector_h
#define TrackingAnalysis_MuonPSimHitSelector_h

#include "SimGeneral/TrackingAnalysis/interface/PSimHitSelector.h"

//! MuonPSimHitSelector class
class MuonPSimHitSelector : public PSimHitSelector
{

public:

    //! Constructor by pset.
    /* Creates a MuonPSimHitSelector with association given by pset.

       /param[in] pset with the configuration values
    */
    MuonPSimHitSelector(edm::ParameterSet const & config) : PSimHitSelector(config) {}

    //! Pre-process event information
    virtual void select(PSimHitCollection &, edm::Event const &, edm::EventSetup const &) const;

};

#endif
