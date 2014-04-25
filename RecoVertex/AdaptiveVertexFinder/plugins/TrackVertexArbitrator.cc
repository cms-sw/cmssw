#include <memory>
#include <set>


#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"


#include "FWCore/Framework/interface/stream/EDProducer.h"
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

class TrackVertexArbitrator : public edm::stream::EDProducer<> {
    public:
	TrackVertexArbitrator(const edm::ParameterSet &params);


	virtual void produce(edm::Event &event, const edm::EventSetup &es) override;

    private:
	bool trackFilter(const reco::TrackRef &track) const;

	edm::EDGetTokenT<reco::VertexCollection> token_primaryVertex;
	edm::EDGetTokenT<reco::VertexCollection> token_secondaryVertex;
	edm::EDGetTokenT<reco::TrackCollection>	 token_tracks; 
	edm::EDGetTokenT<reco::BeamSpot> 	 token_beamSpot; 
	TrackVertexArbitration * theArbitrator;
};

TrackVertexArbitrator::TrackVertexArbitrator(const edm::ParameterSet &params)
{
	token_primaryVertex = consumes<reco::VertexCollection>(params.getParameter<edm::InputTag>("primaryVertices"));
	token_secondaryVertex = consumes<reco::VertexCollection>(params.getParameter<edm::InputTag>("secondaryVertices"));
	token_beamSpot = consumes<reco::BeamSpot>(params.getParameter<edm::InputTag>("beamSpot"));
	token_tracks = consumes<reco::TrackCollection>(params.getParameter<edm::InputTag>("tracks"));
	produces<reco::VertexCollection>();
	theArbitrator = new TrackVertexArbitration(params);
}


void TrackVertexArbitrator::produce(edm::Event &event, const edm::EventSetup &es)
{
	using namespace reco;

	edm::Handle<VertexCollection> secondaryVertices;
	event.getByToken(token_secondaryVertex, secondaryVertices);
        VertexCollection theSecVertexColl = *(secondaryVertices.product());

        edm::Handle<VertexCollection> primaryVertices;
	event.getByToken(token_primaryVertex, primaryVertices);

	std::auto_ptr<VertexCollection> recoVertices(new VertexCollection);
        if(primaryVertices->size()!=0){ 
        const reco::Vertex &pv = (*primaryVertices)[0];
    
        edm::Handle<TrackCollection> tracks;
	event.getByToken(token_tracks, tracks);

        edm::ESHandle<TransientTrackBuilder> trackBuilder;
        es.get<TransientTrackRecord>().get("TransientTrackBuilder",
                                           trackBuilder);

        edm::Handle<BeamSpot> beamSpot;
	event.getByToken(token_beamSpot,beamSpot);

        
	edm::RefVector< TrackCollection >  selectedTracks;
	for(TrackCollection::const_iterator track = tracks->begin();
	    track != tracks->end(); ++track) {
		TrackRef ref(tracks, track - tracks->begin());
		selectedTracks.push_back(ref);
	   
	}
	
	
        const edm::RefVector< TrackCollection > tracksForArbitration= selectedTracks;
	reco::VertexCollection  theRecoVertices = theArbitrator->trackVertexArbitrator(beamSpot, pv, trackBuilder, tracksForArbitration,
	theSecVertexColl);
	
        for(unsigned int ivtx=0; ivtx < theRecoVertices.size(); ivtx++){
         recoVertices->push_back(theRecoVertices[ivtx]);
        }

        }	
	event.put(recoVertices);
	
	
	
}

DEFINE_FWK_MODULE(TrackVertexArbitrator);
