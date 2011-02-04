#include "Validation/RecoMuon/plugins/RecoMuonFromPFProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

using namespace std;

using namespace boost;

using namespace edm;

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"



RecoMuonFromPFProducer::RecoMuonFromPFProducer(const edm::ParameterSet& iConfig) {

  inputTagPF_ 
    = iConfig.getParameter<InputTag>("particles");

  verbose_ = 
    iConfig.getUntrackedParameter<bool>("verbose",false);

  // register products
  produces<reco::MuonCollection>();
}



RecoMuonFromPFProducer::~RecoMuonFromPFProducer() {}


void 
RecoMuonFromPFProducer::beginJob() {}

void 
RecoMuonFromPFProducer::beginRun(edm::Run & run, 
		     const edm::EventSetup & es) {}



void 
RecoMuonFromPFProducer::produce(Event& iEvent, 
		    const EventSetup& iSetup) {


  std::auto_ptr< reco::MuonCollection > 
    pOutput( new reco::MuonCollection); 

  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  iEvent.getByLabel( inputTagPF_, pfCandidates);

  typedef reco::PFCandidateCollection::const_iterator IP;
  for (IP ip=pfCandidates->begin(); ip !=pfCandidates->end(); ++ip ) {

    const reco::PFCandidate& cand = *ip;
    if( cand.particleId() == reco::PFCandidate::mu) {
      if( !cand.muonRef().isAvailable() ) {
	cout<<cand.muonRef().id()<<endl;
	LogError("RecoMuonFromPFProducer")<<"reference to reco::Muon not available for muon PFCandidate "<<cand<<endl;
	
	assert( false );
      }

      pOutput->push_back( *(cand.muonRef()) );
    }
  }

  iEvent.put(pOutput);

}

DEFINE_FWK_MODULE(RecoMuonFromPFProducer);
