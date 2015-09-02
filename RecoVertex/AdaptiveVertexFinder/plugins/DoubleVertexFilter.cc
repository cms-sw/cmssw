#include <memory>
#include <set>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

class DoubleVertexFilter : public edm::global::EDProducer<> {
    public:
	DoubleVertexFilter(const edm::ParameterSet &params);

	virtual void produce(edm::StreamID, edm::Event &event, const edm::EventSetup &es) const override;

    private:
	bool trackFilter(const reco::TrackRef &track) const;

	edm::EDGetTokenT<reco::VertexCollection> token_primaryVertex;
	edm::EDGetTokenT<reco::VertexCollection> token_secondaryVertex;
	double					maxFraction;
};

DoubleVertexFilter::DoubleVertexFilter(const edm::ParameterSet &params) :
	maxFraction(params.getParameter<double>("maxFraction"))
{
	token_primaryVertex = consumes<reco::VertexCollection>(params.getParameter<edm::InputTag>("primaryVertices"));
	token_secondaryVertex = consumes<reco::VertexCollection>(params.getParameter<edm::InputTag>("secondaryVertices"));
	produces<reco::VertexCollection>();
}

static double computeSharedTracks(const reco::Vertex &pv,
                                        const reco::Vertex &sv)
{
	std::set<reco::TrackRef> pvTracks;
	for(std::vector<reco::TrackBaseRef>::const_iterator iter = pv.tracks_begin();
	    iter != pv.tracks_end(); iter++) {
		if (pv.trackWeight(*iter) >= 0.5)
			pvTracks.insert(iter->castTo<reco::TrackRef>());
	}

	unsigned int count = 0, total = 0;
	for(std::vector<reco::TrackBaseRef>::const_iterator iter = sv.tracks_begin();
	    iter != sv.tracks_end(); iter++) {
		if (sv.trackWeight(*iter) >= 0.5) {
			total++;
			count += pvTracks.count(iter->castTo<reco::TrackRef>());
		}
	}

	return (double)count / (double)total;
}

void DoubleVertexFilter::produce(edm::StreamID, edm::Event &event, const edm::EventSetup &es) const
{
	using namespace reco;

	edm::Handle<VertexCollection> primaryVertices;
	event.getByToken(token_primaryVertex, primaryVertices);

	edm::Handle<VertexCollection> secondaryVertices;
	event.getByToken(token_secondaryVertex, secondaryVertices);

	std::vector<reco::Vertex>::const_iterator pv = primaryVertices->begin();

	std::auto_ptr<VertexCollection> recoVertices(new VertexCollection);
	for(std::vector<reco::Vertex>::const_iterator sv = secondaryVertices->begin();
	    sv != secondaryVertices->end(); ++sv) {
		if (computeSharedTracks(*pv, *sv) > maxFraction)
			continue;

		recoVertices->push_back(*sv);
	}

	event.put(recoVertices);
}

DEFINE_FWK_MODULE(DoubleVertexFilter);
