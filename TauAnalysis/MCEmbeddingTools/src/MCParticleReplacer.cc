#include "TauAnalysis/MCEmbeddingTools/interface/MCParticleReplacer.h"
#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerFactory.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "FWCore/Framework/interface/Event.h"

#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"

// replacementMode =
//	0 - remove Myons from existing HepMCProduct and implant taus (+decay products)
//	1 - build new HepMCProduct only with taus (+decay products)
MCParticleReplacer::MCParticleReplacer(const edm::ParameterSet& pset):
  src_(pset.getParameter<edm::InputTag>("src")),
  srcHepMC_(pset.getParameter<edm::InputTag>("hepMcSrc")),
  hepMcMode_(stringToHepMcMode(pset.getParameter<std::string>("hepMcMode"))),
  replacer_(ParticleReplacerFactory::create(pset.getParameter<std::string>("algorithm"), pset)) {

  produces<edm::HepMCProduct>();
  produces<GenFilterInfo>("minVisPtFilter");
}

MCParticleReplacer::~MCParticleReplacer()
{}

MCParticleReplacer::HepMcMode MCParticleReplacer::stringToHepMcMode(const std::string& name) {
  if(name == "new")
    return kNew;
  else if(name == "replace")
    return kReplace;
  else
    throw cms::Exception("Configuration") << "Unsupported hepMcMode " << name << ", should be 'new' or 'replace'" << std::endl;
}

// ------------ method called to produce the data  ------------
void
MCParticleReplacer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
	std::vector<reco::Muon> muons;
	
	edm::Handle< std::vector< reco::CompositeCandidate > > combCandidatesHandle;
	if (iEvent.getByLabel(src_, combCandidatesHandle))
	{
		if (combCandidatesHandle->size()==0)
			return;
		for (size_t idx = 0; idx < combCandidatesHandle->at(0).numberOfDaughters(); ++idx)
		{
			int charge = combCandidatesHandle->at(0).daughter(idx)->charge();
			reco::Particle::LorentzVector p4 = combCandidatesHandle->at(0).daughter(idx)->p4();
			reco::Particle::Point vtx = combCandidatesHandle->at(0).daughter(idx)->vertex();
			muons.push_back( reco::Muon(charge, p4, vtx));
		}
	}
	else
	{
		edm::Handle< edm::View<reco::Candidate> > candsHandle;
		if (iEvent.getByLabel(src_, candsHandle))
		{
			for (size_t idx = 0; idx < candsHandle->size(); ++idx)
			{
				int charge = candsHandle->at(idx).charge();
				reco::Particle::LorentzVector p4 = candsHandle->at(idx).p4();
				reco::Particle::Point vtx = candsHandle->at(idx).vertex();
				muons.push_back( reco::Muon(charge, p4, vtx));
			}
		}
	}
	if (muons.size() == 0)
	{
		edm::LogError("MCParticleReplacer") << "No candidates or muons found!";
		return;
	}
	
  std::auto_ptr<HepMC::GenEvent> evt;
  if(hepMcMode_ == kReplace) {
    edm::Handle<edm::HepMCProduct> HepMCHandle;	 
    iEvent.getByLabel(srcHepMC_, HepMCHandle);

    evt = replacer_->produce(muons, 0, HepMCHandle->GetEvent());
  }
  else if(hepMcMode_ == kNew) {
    evt = replacer_->produce(muons);
  }
  else
    throw cms::Exception("LogicError") << "Invalid hepMcMode " << hepMcMode_ << std::endl;


  if(evt.get() != 0) {
    std::auto_ptr<edm::HepMCProduct> bare_product(new edm::HepMCProduct());  
    bare_product->addHepMCData(evt.release()); // transfer ownership of the HepMC:GenEvent to bare_product

    iEvent.put(bare_product);

    std::auto_ptr<GenFilterInfo> info(new GenFilterInfo(replacer_->tried, replacer_->passed));
    iEvent.put(info, std::string("minVisPtFilter"));
  }
  
}

void MCParticleReplacer::beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup)
{
  replacer_->beginRun(iRun, iSetup);
}

void MCParticleReplacer::endRun(const edm::Run& iRun,const edm::EventSetup& iSetup)
{
  replacer_->endRun();
}

// ------------ method called once each job just before starting event loop  ------------
void 
MCParticleReplacer::beginJob()
{
  replacer_->beginJob();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MCParticleReplacer::endJob()
{
  replacer_->endJob();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MCParticleReplacer);
