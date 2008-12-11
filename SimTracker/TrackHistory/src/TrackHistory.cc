
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
}


void TrackHistory::newEvent (
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
    edm::ESHandle<TrackAssociatorBase> associator;
    setup.get<TrackAssociatorRecord>().get(trackAssociator_, associator);

    // Get the map between recotracks and tp
    association_ = associator->associateRecoToSim (trackCollection, TPCollection, &event);
}


bool TrackHistory::evaluate ( reco::TrackBaseRef tr )
{
    TrackingParticleRef tpr( match(tr) );

    if ( !tpr.isNull() )
    {
        HistoryBase::evaluate(tpr);
        return true;
    }

    return false;
}


TrackingParticleRef TrackHistory::match ( reco::TrackBaseRef tr )
{
    TrackingParticleRef tpr;

    reco::RecoToSimCollection::const_iterator pos = association_.find(tr);
    if (pos == association_.end())
        return tpr;

    const std::vector<std::pair<TrackingParticleRef, double> > &tp = pos->val;

    double m = bestMatchByMaxValue_ ? -1e30 : 1e30;

    for (std::size_t i = 0; i < tp.size(); i++)
    {
        if (bestMatchByMaxValue_ ? (tp[i].second > m) : (tp[i].second < m))
        {
            tpr = tp[i].first;
            m = tp[i].second;
        }
    }

    return tpr;
}
