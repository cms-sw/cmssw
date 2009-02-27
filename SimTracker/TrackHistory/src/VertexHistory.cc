
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
}


void VertexHistory::newEvent (
    const edm::Event & event, const edm::EventSetup & setup
)
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

    // Get the map between recotracks and tp
    reco::RecoToSimCollection
    trackAssociation = trackAssociator->associateRecoToSim (trackCollection, TPCollection, &event);

    // Vertex collection
    edm::Handle<reco::VertexCollection> vertexCollection;
    event.getByLabel(vertexProducer_, vertexCollection);

    // Tracking particle information
    edm::Handle<TrackingVertexCollection>  TVCollection;
    event.getByLabel(trackingTruth_, TVCollection);

    // Get the track associator
    edm::ESHandle<VertexAssociatorBase> vertexAssociator;
    setup.get<VertexAssociatorRecord>().get(vertexAssociator_, vertexAssociator);

    // Get the map between recovertex and tp
    association_ = vertexAssociator->associateRecoToSim (vertexCollection, TVCollection, event, trackAssociation);
}


bool VertexHistory::evaluate ( reco::VertexRef tv )
{
    TrackingVertexRef tvr( match(tv) );

    if ( !tvr.isNull() )
    {
        HistoryBase::evaluate(tvr);
        return true;
    }

    return false;
}


TrackingVertexRef VertexHistory::match ( reco::VertexRef vr )
{
    TrackingVertexRef tvr;

    reco::VertexRecoToSimCollection::const_iterator pos = association_.find(vr);

    if (pos == association_.end()) return tvr;

    const std::vector<std::pair<TrackingVertexRef, double> > &tv = pos->val;

    double m = bestMatchByMaxValue_ ? -1e30 : 1e30;

    for (std::size_t i = 0; i < tv.size(); i++)
    {
        if (bestMatchByMaxValue_ ? (tv[i].second > m) : (tv[i].second < m))
        {
            tvr = tv[i].first;
            m = tv[i].second;
        }
    }

    return tvr;
}
