
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackHistory/interface/TrackHistory.h"


TrackHistory::TrackHistory (
    const edm::ParameterSet & config
) : HistoryBase()
{
    // Name of the track collection
    trackProducer_ = config.getUntrackedParameter<edm::InputTag> ( "trackProducer" );

    // Name of the traking pariticle collection
    trackingTruth_ = config.getUntrackedParameter<edm::InputTag> ( "trackingTruth" );

    // Track association record
    trackAssociator_ = config.getUntrackedParameter<std::string> ( "trackAssociator" );

    // Association by max. value
    bestMatchByMaxValue_ = config.getUntrackedParameter<bool> ( "bestMatchByMaxValue" );

    // Enable RecoToSim association
    enableRecoToSim_ = config.getUntrackedParameter<bool> ( "enableRecoToSim" );

    // Enable SimToReco association
    enableSimToReco_ = config.getUntrackedParameter<bool> ( "enableSimToReco" );

    quality_ = 0.;
}


void TrackHistory::newEvent (
    const edm::Event & event, const edm::EventSetup & setup
)
{
    if ( enableRecoToSim_ || enableSimToReco_ )
    {
        // Track collection
        edm::Handle<edm::View<reco::Track> > trackCollection;
        event.getByLabel(trackProducer_, trackCollection);

        // Tracking particle information
        edm::Handle<TrackingParticleCollection>  TPCollection;
        event.getByLabel(trackingTruth_, TPCollection);

        // Get the track associator
        edm::ESHandle<TrackAssociatorBase> associator;
        setup.get<TrackAssociatorRecord>().get(trackAssociator_, associator);

        // Calculate the map between recotracks -> tp
        if ( enableRecoToSim_ ) recoToSim_ = associator->associateRecoToSim(trackCollection, TPCollection, &event);

        // Calculate the map between recotracks <- tp
        if ( enableSimToReco_ ) simToReco_ = associator->associateSimToReco(trackCollection, TPCollection, &event);
    }
}


bool TrackHistory::evaluate ( reco::TrackBaseRef tr )
{
    if ( !enableRecoToSim_ ) return false;

    std::pair<TrackingParticleRef, double> result =  match(tr, recoToSim_, bestMatchByMaxValue_); 

    TrackingParticleRef tpr( result.first );
    quality_ = result.second;

    if ( !tpr.isNull() )
    {
        HistoryBase::evaluate(tpr);

        recotrack_ = tr;

        return true;
    }

    return false;
}


