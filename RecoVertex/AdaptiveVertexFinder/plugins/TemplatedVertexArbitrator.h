#ifndef TemplatedVertexArbitrator_h
#define TemplatedVertexArbitrator_h
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
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "RecoVertex/AdaptiveVertexFinder/interface/TTHelpers.h"

//#define VTXDEBUG

template <class InputContainer, class VTX>
class TemplatedVertexArbitrator : public edm::stream::EDProducer<> {
    public:
	typedef std::vector<VTX> Product;
	TemplatedVertexArbitrator(const edm::ParameterSet &params); 


	virtual void produce(edm::Event &event, const edm::EventSetup &es) override ;

    private:
	bool trackFilter(const reco::TrackRef &track) const;

	edm::EDGetTokenT<reco::VertexCollection> token_primaryVertex;
	edm::EDGetTokenT<Product> token_secondaryVertex;
	edm::EDGetTokenT<InputContainer>	 token_tracks; 
	edm::EDGetTokenT<reco::BeamSpot> 	 token_beamSpot; 
	TrackVertexArbitration<VTX> * theArbitrator;
};


template <class InputContainer, class VTX>
TemplatedVertexArbitrator<InputContainer,VTX>::TemplatedVertexArbitrator(const edm::ParameterSet &params)
{
	token_primaryVertex = consumes<reco::VertexCollection>(params.getParameter<edm::InputTag>("primaryVertices"));
	token_secondaryVertex = consumes<Product>(params.getParameter<edm::InputTag>("secondaryVertices"));
	token_beamSpot = consumes<reco::BeamSpot>(params.getParameter<edm::InputTag>("beamSpot"));
	token_tracks = consumes<InputContainer>(params.getParameter<edm::InputTag>("tracks"));
	produces<Product>();
	theArbitrator = new TrackVertexArbitration<VTX>(params);
}

template <class InputContainer, class VTX>
void TemplatedVertexArbitrator<InputContainer,VTX>::produce(edm::Event &event, const edm::EventSetup &es)
{
	using namespace reco;

	edm::Handle<Product> secondaryVertices;
	event.getByToken(token_secondaryVertex, secondaryVertices);
	Product theSecVertexColl = *(secondaryVertices.product());

	edm::Handle<VertexCollection> primaryVertices;
	event.getByToken(token_primaryVertex, primaryVertices);

	std::auto_ptr<Product> recoVertices(new Product);
	if(primaryVertices->size()!=0){ 
		const reco::Vertex &pv = (*primaryVertices)[0];

		edm::Handle<InputContainer> tracks;
		event.getByToken(token_tracks, tracks);

		edm::ESHandle<TransientTrackBuilder> trackBuilder;
		es.get<TransientTrackRecord>().get("TransientTrackBuilder",
				trackBuilder);

		edm::Handle<BeamSpot> beamSpot;
		event.getByToken(token_beamSpot,beamSpot);

		std::vector<TransientTrack> selectedTracks;
		for(typename InputContainer::const_iterator track = tracks->begin();
				track != tracks->end(); ++track) {
			 selectedTracks.push_back(tthelpers::buildTT(tracks,trackBuilder,track - tracks->begin()));
		}


		//        const edm::RefVector< TrackCollection > tracksForArbitration= selectedTracks;
		Product  theRecoVertices = theArbitrator->trackVertexArbitrator(beamSpot, pv, selectedTracks,
				theSecVertexColl);

		for(unsigned int ivtx=0; ivtx < theRecoVertices.size(); ivtx++){
			recoVertices->push_back(theRecoVertices[ivtx]);
		}

	}	
	event.put(recoVertices);



}



#endif
