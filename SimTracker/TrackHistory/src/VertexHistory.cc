
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/Records/interface/VertexAssociatorRecord.h"
#include "SimTracker/TrackHistory/interface/VertexHistory.h"

VertexHistory::VertexHistory (
    const edm::ParameterSet & config
) : HistoryBase()
{
    // Name of the track collection
    trackProducer_ = config.getUntrackedParameter<edm::InputTag> ( "trackProducer" );

    // Name of the track collection
    vertexProducer_ = config.getUntrackedParameter<edm::InputTag> ( "vertexProducer" );

    // Name of the traking pariticle collection
    trackingTruth_ = config.getUntrackedParameter<edm::InputTag> ( "trackingTruth" );

    // Track association record
    trackAssociator_ = config.getUntrackedParameter<std::string> ( "trackAssociator" );

    // Track association record
    vertexAssociator_ = config.getUntrackedParameter<std::string> ( "vertexAssociator" );

    // Association by max. value
    bestMatchByMaxValue_ = config.getUntrackedParameter<bool> ( "bestMatchByMaxValue" );

    // Enable RecoToSim association
    enableRecoToSim_ = config.getUntrackedParameter<bool> ( "enableRecoToSim" );

    // Enable SimToReco association
    enableSimToReco_ = config.getUntrackedParameter<bool> ( "enableSimToReco" );

    quality_ = 0.;
}


void VertexHistory::newEvent (
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
        edm::ESHandle<TrackAssociatorBase> trackAssociator;
        setup.get<TrackAssociatorRecord>().get(trackAssociator_, trackAssociator);

        // Vertex collection
        edm::Handle<edm::View<reco::Vertex> > vertexCollection;
        event.getByLabel(vertexProducer_, vertexCollection);

        // Tracking particle information
        edm::Handle<TrackingVertexCollection>  TVCollection;
        event.getByLabel(trackingTruth_, TVCollection);

        // Get the track associator
        edm::ESHandle<VertexAssociatorBase> vertexAssociator;
        setup.get<VertexAssociatorRecord>().get(vertexAssociator_, vertexAssociator);

        if ( enableRecoToSim_ )
        {
            // Get the map between recovertex -> simvertex
            reco::RecoToSimCollection
	      trackRecoToSim = trackAssociator->associateRecoToSim(trackCollection, TPCollection, &event,&setup);

            // Calculate the map between recovertex -> simvertex
            recoToSim_ = vertexAssociator->associateRecoToSim(vertexCollection, TVCollection, event, trackRecoToSim);
        }

        if ( enableSimToReco_ )
        {
            // Get the map between recovertex <- simvertex
            reco::SimToRecoCollection
	      trackSimToReco = trackAssociator->associateSimToReco (trackCollection, TPCollection, &event, &setup);

            // Calculate the map between recovertex <- simvertex
            simToReco_ = vertexAssociator->associateSimToReco(vertexCollection, TVCollection, event, trackSimToReco);
        }

    }
}


bool VertexHistory::evaluate (reco::VertexBaseRef tv)
{

    if ( !enableRecoToSim_ ) return false;

    std::pair<TrackingVertexRef, double> result =  match(tv, recoToSim_, bestMatchByMaxValue_);

    TrackingVertexRef tvr( result.first );
    quality_ = result.second;

    if ( !tvr.isNull() )
    {
        HistoryBase::evaluate(tvr);

        recovertex_ = tv;

        return true;
    }

    return false;
}

