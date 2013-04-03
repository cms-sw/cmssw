#ifndef TrackingAnalysis_PSimHitSelector_h
#define TrackingAnalysis_PSimHitSelector_h

#include <map>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

//! PSimHitSelector class
class PSimHitSelector
{

public:

    typedef std::vector<PSimHit> PSimHitCollection;

    //! Constructor by pset.
    /* Creates a MuonPSimHitSelector with association given by pset.

       /param[in] pset with the configuration values
    */
    PSimHitSelector(edm::ParameterSet const &);
    std::string mixLabel_;

    //! Virtual destructor.
    virtual ~PSimHitSelector() {}

    //! Select the psimhit add them to a PSimHitCollection
    virtual void select(PSimHitCollection &, edm::Event const &, edm::EventSetup const &) const;

protected:

    typedef std::map<std::string, std::vector<std::string> > PSimHitCollectionMap;

    PSimHitCollectionMap pSimHitCollectionMap_;
};

#endif
