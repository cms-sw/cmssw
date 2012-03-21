#include <memory>
#include <set>


#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackCompatibilityEstimator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexSmoother.h"
#include "DataFormats/Math/interface/deltaR.h"


#include "RecoVertex/ConfigurableVertexReco/interface/ConfigurableVertexReconstructor.h"
#include "RecoVertex/AdaptiveVertexFinder/interface/TrackVertexArbitratration.h"



//#define VTXDEBUG

class TrackVertexArbitrator : public edm::EDProducer {
    public:
	TrackVertexArbitrator(const edm::ParameterSet &params);


	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	bool trackFilter(const reco::TrackRef &track) const;

	edm::InputTag				primaryVertexCollection;
	edm::InputTag				secondaryVertexCollection;
	edm::InputTag				trackCollection;
        edm::InputTag                           beamSpotCollection;
	TrackVertexArbitration * theArbitrator;
};

TrackVertexArbitrator::TrackVertexArbitrator(const edm::ParameterSet &params) :
	primaryVertexCollection      (params.getParameter<edm::InputTag>("primaryVertices")),
	secondaryVertexCollection    (params.getParameter<edm::InputTag>("secondaryVertices")),
	trackCollection              (params.getParameter<edm::InputTag>("tracks")),
        beamSpotCollection           (params.getParameter<edm::InputTag>("beamSpot"))
{
	produces<reco::VertexCollection>();
	theArbitrator = new TrackVertexArbitration(params);
}


void TrackVertexArbitrator::produce(edm::Event &event, const edm::EventSetup &es)
{
	using namespace reco;

	edm::Handle<VertexCollection> secondaryVertices;
	event.getByLabel(secondaryVertexCollection, secondaryVertices);
        VertexCollection theSecVertexColl = *(secondaryVertices.product());

        edm::Handle<VertexCollection> primaryVertices;
        event.getByLabel(primaryVertexCollection, primaryVertices);
        const reco::Vertex &pv = (*primaryVertices)[0];

        edm::Handle<TrackCollection> tracks;
        event.getByLabel(trackCollection, tracks);

        edm::ESHandle<TransientTrackBuilder> trackBuilder;
        es.get<TransientTrackRecord>().get("TransientTrackBuilder",
                                           trackBuilder);

        edm::Handle<BeamSpot> beamSpot;
        event.getByLabel(beamSpotCollection, beamSpot);

        
	edm::RefVector< TrackCollection >  selectedTracks;
	for(TrackCollection::const_iterator track = tracks->begin();
	    track != tracks->end(); ++track) {
		TrackRef ref(tracks, track - tracks->begin());
		selectedTracks.push_back(ref);
	   
	}
	
	
        const edm::RefVector< TrackCollection > tracksForArbitration= selectedTracks;
	reco::VertexCollection  theRecoVertices = theArbitrator->trackVertexArbitrator(beamSpot, pv, trackBuilder, tracksForArbitration,
	theSecVertexColl);
	
	std::auto_ptr<VertexCollection> recoVertices(new VertexCollection);
        for(unsigned int ivtx=0; ivtx < theRecoVertices.size(); ivtx++){
         recoVertices->push_back(theRecoVertices[ivtx]);
        }

	
	event.put(recoVertices);
	
	
	
}

DEFINE_FWK_MODULE(TrackVertexArbitrator);
