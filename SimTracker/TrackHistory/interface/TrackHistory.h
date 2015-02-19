#ifndef TrackHistory_h
#define TrackHistory_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimTracker/TrackHistory/interface/HistoryBase.h"
#include "SimTracker/TrackHistory/interface/Utils.h"

//! This class traces the simulated and generated history of a given track.
class TrackHistory : public HistoryBase
{

public:

    //! Constructor by pset.
    /* Creates a TrackHistory with association given by pset.

       /param[in] pset with the configuration values
    */
    TrackHistory(const edm::ParameterSet &,
                 edm::ConsumesCollector&& );

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
        if ( enableSimToReco_ )
        {
            std::pair<reco::TrackBaseRef, double> result =  match(tpr, simToReco_, bestMatchByMaxValue_);
            recotrack_ = result.first;
            quality_ =  result.second;
        }
        return HistoryBase::evaluate(tpr);
    }


    //! Evaluate reco::Track history using a given association.
    /* Return false when the track association is not possible (fake track).

       /param[in] TrackRef to a reco::track
       /param[out] boolean that is false when a fake track is detected
    */
    bool evaluate (reco::TrackBaseRef);

    //! Return a reference to the reconstructed track.
    const reco::TrackBaseRef & recoTrack() const
    {
        return recotrack_;
    }

    double quality() const
    {
        return quality_;
    }

private:

    bool newEvent_;

    bool bestMatchByMaxValue_;

    bool enableRecoToSim_, enableSimToReco_;

    double quality_;

    edm::InputTag trackProducer_;

    edm::InputTag trackingTruth_;

    edm::InputTag trackAssociator_;

    reco::TrackBaseRef recotrack_;

    reco::RecoToSimCollection recoToSim_;

    reco::SimToRecoCollection simToReco_;

};

#endif
