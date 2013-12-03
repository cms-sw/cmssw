// system include files

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// #include <list>
#include <vector>
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "Math/VectorUtil.h"



class SelectReplacementCandidates : public edm::EDProducer {
public:
	SelectReplacementCandidates(const edm::ParameterSet& iSetup);
	~SelectReplacementCandidates();

	virtual void produce(edm::Event& iEvent, const edm::EventSetup& iConfig);
	virtual void beginJob();
	virtual void endJob();

private:
	// ----------member functions ---------------------------
	void getRawIDsAdvanced(const edm::Event& iEvent, const edm::EventSetup& iConfig, std::vector<uint32_t> * L, reco::Muon * muon, bool includeHCAL);
	int determineMuonsToUse(const edm::Event& iEvent, const edm::EventSetup& iConfig, reco::Muon * muon1, reco::Muon * muon2);
	int determineMuonsToUse_old(const edm::Event& iEvent, const edm::EventSetup& iConfig, reco::Muon * muon1, reco::Muon * muon2);
	template < typename T > void ProductNotFound(const edm::Event& iEvent, edm::InputTag inputTag);
	void transformMuMu2TauTau(reco::Muon * muon1, reco::Muon * muon2);
	
	// ----------member data ---------------------------
	TrackDetectorAssociator trackAssociator_;
	TrackAssociatorParameters parameters_;
	edm::InputTag muonInputTag_;
	
	double targetParticleMass_;
	int targetParticlePdgID_;	
};

