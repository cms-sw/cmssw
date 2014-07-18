#include "TauAnalysis/MCEmbeddingTools/plugins/SelectReplacementCandidates.h"
#include "DataFormats/Math/interface/deltaR.h"

SelectReplacementCandidates::SelectReplacementCandidates(const edm::ParameterSet& iConfig)
{
	using namespace edm;
	using namespace std;
	produces< vector<uint32_t> >();
	produces< vector<uint32_t> >("assocHitsWithHCAL");
	produces< std::vector<reco::Muon> >();
	
	targetParticleMass_=iConfig.getUntrackedParameter<double>("targetParticlesMass",1.77690);
	targetParticlePdgID_=iConfig.getUntrackedParameter<int>("targetParticlesPdgID",15);
	
	muonInputTag_ = iConfig.getParameter<edm::InputTag>("muonInputTag");
	edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
	edm::ConsumesCollector iC = consumesCollector();
	parameters_.loadParameters( parameters, iC );
	
	trackAssociator_.useDefaultPropagator();
}

SelectReplacementCandidates::~SelectReplacementCandidates()
{

}

void SelectReplacementCandidates::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	using namespace std;
	using namespace edm;
	reco::Muon muon1, muon2;

	// determine the muons to be used
	// these can be reconstructed muons or muons from a HepMC::Event
	if (determineMuonsToUse(iEvent, iSetup, &muon1, &muon2)!=0)
		return;
	
	vector<uint32_t> hits ;	
	getRawIDsAdvanced(iEvent, iSetup, &hits, &muon1, false);
	getRawIDsAdvanced(iEvent, iSetup, &hits, &muon2, false);
	std::auto_ptr< vector<uint32_t> > selectedHitsAutoPtr(new vector<uint32_t>(hits) );
	iEvent.put( selectedHitsAutoPtr );

	vector<uint32_t> assoc_hits_withHCAL;   
	getRawIDsAdvanced(iEvent, iSetup, &assoc_hits_withHCAL, &muon1, true);
	getRawIDsAdvanced(iEvent, iSetup, &assoc_hits_withHCAL, &muon2, true);
	std::cout << "found in total " << assoc_hits_withHCAL.size() << " cells with associated hits with hcal\n";
	std::auto_ptr< vector<uint32_t> > selectedAssocHitsWithHCALAutoPtr(new vector<uint32_t>(assoc_hits_withHCAL) );
	iEvent.put( selectedAssocHitsWithHCALAutoPtr, "assocHitsWithHCAL");
                                                	
	std::vector<reco::Muon> muons;
	transformMuMu2TauTau(&muon1, &muon2);
	muons.push_back(muon1);
	muons.push_back(muon2);
	std::auto_ptr< std::vector<reco::Muon> > selectedMuonsPtr(new std::vector<reco::Muon>(muons) );
	iEvent.put( selectedMuonsPtr );

}

void SelectReplacementCandidates::beginJob()
{

}

void SelectReplacementCandidates::endJob()
{

}


int SelectReplacementCandidates::determineMuonsToUse(const edm::Event& iEvent, const edm::EventSetup& iConfig, reco::Muon * muon1, reco::Muon * muon2)
{
	using namespace edm;
	using namespace reco;
	using namespace std;

	Handle< edm::RefToBaseVector<reco::Candidate> > zCandidate_handle;
	if (!iEvent.getByLabel("dimuonsGlobal", zCandidate_handle))
	{
		std::cout << "Could not find product: " << "dimuonsGlobal" << "\n";    
    std::vector< edm::Handle< edm::RefToBaseVector<reco::Candidate> >  > allHandles;
    iEvent.getManyByType(allHandles);
    std::vector< edm::Handle< edm::RefToBaseVector<reco::Candidate> > >::iterator it;
    for (it = allHandles.begin(); it != allHandles.end(); it++)
    {
      std::cout << "available product: " << (*it).provenance()->moduleLabel() << ", " << (*it).provenance()->productInstanceName() << ", " << (*it).provenance()->processName();
    }

	  std::cout << "Objekt nicht gefunden: dimuonsGloal\n";
	  return -1;
	}
//	std::cout << zCandidate_handle->size() << " Kandidaten gefunden!\n";

	unsigned int nMuons = zCandidate_handle->size();
	if (nMuons==0)
		return -1;

	for (edm::RefToBaseVector<reco::Candidate>::const_iterator z = zCandidate_handle->begin(); z!=zCandidate_handle->end(); ++z)
	{
		reco::Particle::LorentzVector muon1p4 = z->get()->daughter(0)->p4();
		reco::Particle::LorentzVector muon2p4 = z->get()->daughter(1)->p4();

		edm::Handle<edm::View<reco::Muon> > trackCollection;
		iEvent.getByLabel(muonInputTag_, trackCollection);
		const edm::View<reco::Muon>& muons = * trackCollection;

		bool found1=false, found2=false;
		for (unsigned int i=0;i<muons.size() && !(found1 && found2);i++)
		{
			if (deltaR(muon1p4,muons[i].p4())<0.1)
			{
				*muon1 = muons[i];
				found1=true;
			}
			if (deltaR(muon2p4,muons[i].p4())<0.1)
			{
				*muon2 = muons[i];
				found2=true;
			}
		}

		break;
	}
	return 0;
}


int SelectReplacementCandidates::determineMuonsToUse_old(const edm::Event& iEvent, const edm::EventSetup& iConfig, reco::Muon * muon1, reco::Muon * muon2)
{
    using namespace edm;
    using namespace reco;
    using namespace std;

	Handle<MuonCollection> muonHandle;
	if (!iEvent.getByLabel(muonInputTag_,muonHandle))
		ProductNotFound<reco::MuonCollection>(iEvent, muonInputTag_);
	
	edm::Handle<edm::View<reco::Muon> > trackCollection;
	iEvent.getByLabel(muonInputTag_, trackCollection);
	const edm::View<reco::Muon>& muons = * trackCollection;

	unsigned int nMuons = muons.size();
	if (nMuons<2)
		return -1;
	
	*muon1 = muons[0];
	*muon2 = muons[1];
	std::cout << muons[0].p4() << "\n";
	std::cout << muons[1].p4() << "\n";
    
  return 0;
}

void SelectReplacementCandidates::getRawIDsAdvanced(const edm::Event& iEvent, const edm::EventSetup& iConfig, std::vector<uint32_t> * L, reco::Muon * muon, bool includeHCAL)
{
  using namespace edm;
  using namespace reco;
  using namespace std;

	if (muon->bestTrackRef().isNonnull())
	{
		TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iConfig, *(muon->bestTrackRef().get()), parameters_);
		if (includeHCAL)
		{
			for(std::vector<const EcalRecHit*>::const_iterator hit = info.ecalRecHits.begin(); hit != info.ecalRecHits.end(); ++hit)
      	L->push_back((*hit)->detid().rawId());
                                                                                                
			for(std::vector<const HBHERecHit*>::const_iterator hit = info.hcalRecHits.begin(); hit != info.hcalRecHits.end(); ++hit)
				L->push_back((*hit)->detid().rawId());
                                                                                                                                                              }
			int recs = (muon->bestTrackRef().get())->recHitsSize();
			for (int i=0; i<recs; i++)
				L->push_back(((muon->bestTrackRef().get())->recHit(i).get())->geographicalId().rawId());
	}
	else
		std::cout << "ERROR: Muon has no bestTrackRef!!\n";

    std::cout << " with " <<  L->size() << " muon hits found\n";
//     return L;
}


template < typename T > void SelectReplacementCandidates::ProductNotFound(const edm::Event& iEvent, edm::InputTag inputTag)
{
    std::cout << "--- TauHitSelector ------------------------------------\n";
    std::cout << "Could not find product with:\n"<< inputTag << "\n";
    std::vector< edm::Handle< T > > allHandles; 
    iEvent.getManyByType(allHandles);
    typename std::vector< edm::Handle< T > >::iterator it;
    for (it = allHandles.begin(); it != allHandles.end(); it++)
    {
            std::cout << "module label:        " << (*it).provenance()->moduleLabel() << "\n";
            std::cout << "productInstanceName: " << (*it).provenance()->productInstanceName() << "\n";
            std::cout << "processName:         " << (*it).provenance()->processName() << "\n\n";
    }
    std::cout << "-------------------------------------------------------\n";		
		return;
}


///	transform muon into tau
void SelectReplacementCandidates::transformMuMu2TauTau(reco::Muon * muon1, reco::Muon * muon2)
{
	using namespace edm;
	using namespace reco;
	using namespace std;
	
	reco::Particle::LorentzVector muon1_momentum = muon1->p4();
	reco::Particle::LorentzVector muon2_momentum =  muon2->p4();
	reco::Particle::LorentzVector z_momentum = muon1_momentum + muon2_momentum;

	ROOT::Math::Boost booster(z_momentum.BoostToCM());
	ROOT::Math::Boost invbooster(booster.Inverse());
	
	reco::Particle::LorentzVector Zb = booster(z_momentum);

	reco::Particle::LorentzVector muon1b = booster(muon1_momentum);
	reco::Particle::LorentzVector muon2b = booster(muon2_momentum);
	
	double tau_mass2 = targetParticleMass_*targetParticleMass_;

	double muonxb_mom2 = muon1b.x()*muon1b.x() + muon1b.y()*muon1b.y() + muon1b.z() * muon1b.z();
	double tauxb_mom2 = 0.25 * Zb.t() * Zb.t() - tau_mass2;

	float scaling1 = sqrt(tauxb_mom2 / muonxb_mom2);
	float scaling2 = scaling1;

	float tauEnergy= Zb.t() / 2.;

	if (tauEnergy*tauEnergy<tau_mass2)
		return;
	
	reco::Particle::LorentzVector tau1b_mom = reco::Particle::LorentzVector(scaling1*muon1b.x(),scaling1*muon1b.y(),scaling1*muon1b.z(),tauEnergy);
	reco::Particle::LorentzVector tau2b_mom = reco::Particle::LorentzVector(scaling2*muon2b.x(),scaling2*muon2b.y(),scaling2*muon2b.z(),tauEnergy);

	// some checks
	// the following test guarantees a deviation
	// of less than 0.1% for phi and theta for the
	// original muons and the placed taus
	// (in the centre-of-mass system of the z boson)
	assert((muon1b.phi()-tau1b_mom.phi())/muon1b.phi()<0.001);
	assert((muon2b.phi()-tau2b_mom.phi())/muon2b.phi()<0.001);
	assert((muon1b.theta()-tau1b_mom.theta())/muon1b.theta()<0.001);
	assert((muon2b.theta()-tau2b_mom.theta())/muon2b.theta()<0.001);	

	reco::Particle::LorentzVector tau1_mom = (invbooster(tau1b_mom));
	reco::Particle::LorentzVector tau2_mom = (invbooster(tau2b_mom));
	
	// some additional checks
	// the following tests guarantee a deviation of less
	// than 0.1% for the following values of the original
	// muons and the placed taus
	//	invariant mass
	//	transverse momentum
	assert(((muon1_momentum+muon1_momentum).mass()-(tau1_mom+tau2_mom).mass())/(muon1_momentum+muon1_momentum).mass()<0.001);
	assert(((muon1_momentum+muon2_momentum).pt()-(tau1_mom+tau2_mom).pt())/(muon1_momentum+muon1_momentum).pt()<0.001);

	muon1->setP4(tau1_mom);
	muon2->setP4(tau2_mom);

	muon1->setPdgId(targetParticlePdgID_*muon1->pdgId()/abs(muon1->pdgId()));
	muon2->setPdgId(targetParticlePdgID_*muon2->pdgId()/abs(muon2->pdgId()));

	muon1->setStatus(1);
	muon2->setStatus(1);

	return;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(SelectReplacementCandidates);

