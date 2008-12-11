#ifndef TrackHistory_h
#define TrackHistory_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "SimTracker/TrackHistory/interface/HistoryBase.h"

//! This class traces the simulated and generated history of a given track.
class TrackHistory : public HistoryBase
{

public:

    //! Constructor by pset.
    /* Creates a TrackHistory with association given by pset.

       /param[in] pset with the configuration values
    */
    TrackHistory(const edm::ParameterSet &);

    //! Pre-process event information (for accessing reconstruction information)
    void newEvent(const edm::Event &, const edm::EventSetup &);

    //! Evaluate track history using a TrackingParticleRef.
    /* Return false when the history cannot be determined upto a given depth.
       If not depth is pass to the function no restriction are apply to it.

       /param[in] TrackingParticleRef of a simulated track
       /param[in] depth of the track history
       /param[out] boolean that is true when history can be determined
    */
    bool evaluate(TrackingParticleRef tpr)
    {
        return HistoryBase::evaluate(tpr);
    }

    //! Evaluate reco::Track history using a given association.
    /* Return false when the track association is not possible (fake track).

       /param[in] TrackRef to a reco::track
       /param[out] boolean that is false when a fake track is detected
    */
    bool evaluate (reco::TrackBaseRef);

    //! Return the initial tracking particle from the history.
    const TrackingParticleRef & simParticle() const
    {
        return simParticleTrail_[0];
    }

    //! Returns a pointer to most primitive status 1 or 2 particle.
    const HepMC::GenParticle * genParticle() const
    {
        if ( genParticleTrail_.empty() ) return 0;
        return genParticleTrail_[genParticleTrail_.size()-1];
    }

private:

    bool newEvent_;

    bool bestMatchByMaxValue_;

    edm::InputTag trackProducer_;

    edm::InputTag trackingTruth_;

    std::string trackAssociator_;

    reco::RecoToSimCollection association_;

    TrackingParticleRef match ( reco::TrackBaseRef );
};

#endif
