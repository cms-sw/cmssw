#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"

using namespace edm;
using namespace std;

class NvertexFilter : public edm::EDFilter{

  public:

    explicit NvertexFilter(const edm::ParameterSet & iConfig);
    ~NvertexFilter();
    virtual bool filter(edm::Event&, const edm::EventSetup&) override;
    virtual void beginJob();
    virtual void endJob();

	private:
    double minNvtx, maxNvtx;
	 TH1F *Nvtx;

};

NvertexFilter::NvertexFilter(const edm::ParameterSet & iConfig) {

   minNvtx = iConfig.getParameter<double>("minNvtx");
   maxNvtx = iConfig.getParameter<double>("maxNvtx");
 

//	edm::Service<TFileService> fs;
//	Nvtx  = fs->make<TH1F> ("Nvtx" ,"Number of vertices", 80, 0, 80);
 	
}

NvertexFilter::~NvertexFilter() {
}

// ------------ method called on each new Event  ------------
bool NvertexFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {

	Handle<reco::VertexCollection> vertexHandle;
	iEvent.getByLabel("offlinePrimaryVertices", vertexHandle);
   if (! vertexHandle.isValid())
	{
		std::cout << __FUNCTION__ << ":" << __LINE__ << ":vertexHandle handle not found!" << std::endl;
		assert(false);
	}
	const int nvtx = vertexHandle->size();
	bool pass = false;
	if (nvtx >= minNvtx && nvtx <= maxNvtx) pass = true;

//	Nvtx->Fill(nvtx);

	return pass;
}

// ------------ method called once each job just before starting event loop  ------------
void NvertexFilter::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void NvertexFilter::endJob() {
	std::cout << __FUNCTION__ << std::endl;
	std::cout << ">>> Min/Max Nvtx = " << minNvtx << " / " << maxNvtx << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(NvertexFilter);
