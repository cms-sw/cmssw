#ifndef TrackingAnalysis_TrackerPSimHitSelector_h
#define TrackingAnalysis_TrackerPSimHitSelector_h

#include "SimGeneral/TrackingAnalysis/interface/PSimHitSelector.h"

//! TrackerPSimHitSelector class
class TrackerPSimHitSelector : public PSimHitSelector
{

public:

    //! Constructor by pset.
    /* Creates a TrackerPSimHitSelector with association given by pset.

       /param[in] pset with the configuration values
    */
    TrackerPSimHitSelector(edm::ParameterSet const & config) : PSimHitSelector(config) {}

    //! Pre-process event information
    virtual void select(PSimHitCollection &, edm::Event const &, edm::EventSetup const &) const;

};

#endif
