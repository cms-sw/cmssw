#include <memory>
#include <set>

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

class DoubleVertexFilter : public edm::EDProducer {
    public:
	DoubleVertexFilter(const edm::ParameterSet &params);

	virtual void produce(edm::Event &event, const edm::EventSetup &es);

    private:
	bool trackFilter(const reco::TrackRef &track) const;

	edm::InputTag				primaryVertexCollection;
	edm::InputTag				secondaryVertexCollection;
	double					maxFraction;
};

DoubleVertexFilter::DoubleVertexFilter(const edm::ParameterSet &params) :
	primaryVertexCollection(params.getParameter<edm::InputTag>("primaryVertices")),
	secondaryVertexCollection(params.getParameter<edm::InputTag>("secondaryVertices")),
	maxFraction(params.getParameter<double>("maxFraction"))
{
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

void DoubleVertexFilter::produce(edm::Event &event, const edm::EventSetup &es)
{
	using namespace reco;

	edm::Handle<VertexCollection> primaryVertices;
	event.getByLabel(primaryVertexCollection, primaryVertices);

	edm::Handle<VertexCollection> secondaryVertices;
	event.getByLabel(secondaryVertexCollection, secondaryVertices);

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
