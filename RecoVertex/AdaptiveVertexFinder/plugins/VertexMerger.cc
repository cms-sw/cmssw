#include <memory>
#include <set>

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

class VertexMerger : public edm::stream::EDProducer<> {
    public:
	VertexMerger(const edm::ParameterSet &params);

	virtual void produce(edm::Event &event, const edm::EventSetup &es) override;

    private:
	bool trackFilter(const reco::TrackRef &track) const;

	edm::EDGetTokenT<reco::VertexCollection> token_secondaryVertex;
	double					maxFraction;
	double					minSignificance;
};

VertexMerger::VertexMerger(const edm::ParameterSet &params) :
	maxFraction(params.getParameter<double>("maxFraction")),
	minSignificance(params.getParameter<double>("minSignificance"))
{
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

void VertexMerger::produce(edm::Event &event, const edm::EventSetup &es)
{
	using namespace reco;

	edm::Handle<VertexCollection> secondaryVertices;
	event.getByToken(token_secondaryVertex, secondaryVertices);

        VertexDistance3D dist;
	std::auto_ptr<VertexCollection> recoVertices(new VertexCollection);
	for(std::vector<reco::Vertex>::const_iterator sv = secondaryVertices->begin();
	    sv != secondaryVertices->end(); ++sv) {
          recoVertices->push_back(*sv);
        }
       for(std::vector<reco::Vertex>::iterator sv = recoVertices->begin();
	    sv != recoVertices->end(); ++sv) {

        bool shared=false;
       for(std::vector<reco::Vertex>::iterator sv2 = recoVertices->begin();
	    sv2 != recoVertices->end(); ++sv2) {
                  double fr=computeSharedTracks(*sv2, *sv);
        //        std::cout << sv2-recoVertices->begin() << " vs " << sv-recoVertices->begin() << " : " << fr << " "  <<  computeSharedTracks(*sv, *sv2) << " sig " << dist.distance(*sv,*sv2).significance() << std::endl;
          //      std::cout << (fr > maxFraction) << " && " << (dist.distance(*sv,*sv2).significance() < 2)  <<  " && " <<  (sv-sv2!=0)  << " && " <<  (fr >= computeSharedTracks(*sv2, *sv))  << std::endl;
		if (fr > maxFraction && dist.distance(*sv,*sv2).significance() < minSignificance && sv-sv2!=0 
                    && fr >= computeSharedTracks(*sv, *sv2) )
		  {
                      shared=true; 
                     // std::cout << "shared " << sv-recoVertices->begin() << " and "  << sv2-recoVertices->begin() << " fractions: " << fr << " , "  << computeSharedTracks(*sv2, *sv) << " sig: " <<  dist.distance(*sv,*sv2).significance() <<  std::endl;
         
                  }
                 

	}
        if(shared) { sv=recoVertices->erase(sv)-1; }
     //    std::cout << "it = " <<  sv-recoVertices->begin() << " new size is: " << recoVertices->size() <<   std::endl;
       }

	event.put(recoVertices);



}

DEFINE_FWK_MODULE(VertexMerger);
