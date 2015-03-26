
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackHistory/interface/TrackHistory.h"


TrackHistory::TrackHistory (
  const edm::ParameterSet & config,
  edm::ConsumesCollector&& collector
) : HistoryBase()
{
    // Name of the track collection
    trackProducer_ = config.getUntrackedParameter<edm::InputTag> ( "trackProducer" );
    collector.consumes<edm::View<reco::Track> >(trackProducer_);

    // Name of the traking pariticle collection
    trackingTruth_ = config.getUntrackedParameter<edm::InputTag> ( "trackingTruth" );
    collector.consumes<TrackingParticleCollection>(trackingTruth_);

    // Track association record
    trackAssociator_ = config.getUntrackedParameter<edm::InputTag> ( "trackAssociator" );

    // Association by max. value
    bestMatchByMaxValue_ = config.getUntrackedParameter<bool> ( "bestMatchByMaxValue" );

    // Enable RecoToSim association
    enableRecoToSim_ = config.getUntrackedParameter<bool> ( "enableRecoToSim" );

    // Enable SimToReco association
    enableSimToReco_ = config.getUntrackedParameter<bool> ( "enableSimToReco" );

    if(enableRecoToSim_ or enableSimToReco_) {
      collector.consumes<reco::TrackToTrackingParticleAssociator>(trackAssociator_);
    }

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
        edm::Handle<reco::TrackToTrackingParticleAssociator> associator;
        event.getByLabel(trackAssociator_, associator);

        // Calculate the map between recotracks -> tp
        if ( enableRecoToSim_ ) recoToSim_ = associator->associateRecoToSim(trackCollection, TPCollection);

        // Calculate the map between recotracks <- tp
        if ( enableSimToReco_ ) simToReco_ = associator->associateSimToReco(trackCollection, TPCollection );
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


