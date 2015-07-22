
#include "SimTracker/TrackHistory/interface/VertexHistory.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"

VertexHistory::VertexHistory (
  const edm::ParameterSet & config,
  edm::ConsumesCollector&& collector
) : HistoryBase()
{
    // Name of the track collection
    vertexProducer_ = config.getUntrackedParameter<edm::InputTag> ( "vertexProducer" );

    // Name of the traking pariticle collection
    trackingTruth_ = config.getUntrackedParameter<edm::InputTag> ( "trackingTruth" );

    // Vertex association record
    vertexAssociator_ = config.getUntrackedParameter<edm::InputTag> ( "vertexAssociator" );

    // Association by max. value
    bestMatchByMaxValue_ = config.getUntrackedParameter<bool> ( "bestMatchByMaxValue" );

    // Enable RecoToSim association
    enableRecoToSim_ = config.getUntrackedParameter<bool> ( "enableRecoToSim" );

    // Enable SimToReco association
    enableSimToReco_ = config.getUntrackedParameter<bool> ( "enableSimToReco" );

    if(enableRecoToSim_ or enableSimToReco_) {
      collector.consumes<edm::View<reco::Vertex>>(vertexProducer_);
      collector.consumes<TrackingVertexCollection>(trackingTruth_);
      collector.consumes<reco::VertexToTrackingVertexAssociator>(vertexAssociator_);
    }

    quality_ = 0.;
}


void VertexHistory::newEvent (
    const edm::Event & event, const edm::EventSetup & setup
)
{
    if ( enableRecoToSim_ || enableSimToReco_ )
    {
        // Vertex collection
        edm::Handle<edm::View<reco::Vertex> > vertexCollection;
        event.getByLabel(vertexProducer_, vertexCollection);

        // Tracking particle information
        edm::Handle<TrackingVertexCollection>  TVCollection;
        event.getByLabel(trackingTruth_, TVCollection);

        // Get the track associator
        edm::Handle<reco::VertexToTrackingVertexAssociator> vertexAssociator;
        event.getByLabel(vertexAssociator_, vertexAssociator);

        if ( enableRecoToSim_ )
        {
            // Calculate the map between recovertex -> simvertex
            recoToSim_ = vertexAssociator->associateRecoToSim(vertexCollection, TVCollection);
        }

        if ( enableSimToReco_ )
        {
            // Calculate the map between recovertex <- simvertex
            simToReco_ = vertexAssociator->associateSimToReco(vertexCollection, TVCollection);
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

