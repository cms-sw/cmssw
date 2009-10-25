#include "TauAnalysis/MCEmbeddingTools/interface/MCParticleReplacer.h"
#include "TauAnalysis/MCEmbeddingTools/interface/ParticleReplacerFactory.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

// replacementMode =
//	0 - remove Myons from existing HepMCProduct and implant taus (+decay products)
//	1 - build new HepMCProduct only with taus (+decay products)
MCParticleReplacer::MCParticleReplacer(const edm::ParameterSet& pset):
  src_(pset.getParameter<edm::InputTag>("selectedParticles")),
  srcHepMC_(pset.getParameter<edm::InputTag>("HepMCSource")),
  replacementMode_(pset.getParameter<unsigned int>("replacementMode")),
  replacer_(ParticleReplacerFactory::create(pset.getUntrackedParameter<int>("desiredReplacerClass", 1), pset)) {

  produces<edm::HepMCProduct>();
}

MCParticleReplacer::~MCParticleReplacer()
{}

// ------------ method called to produce the data  ------------
void
MCParticleReplacer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<reco::MuonCollection> muons;
  
  if (!iEvent.getByLabel(src_, muons))
  	return;

  std::auto_ptr<HepMC::GenEvent> evt;
  if(replacementMode_ == 0) {
    edm::Handle<edm::HepMCProduct> HepMCHandle;	 
    iEvent.getByLabel(srcHepMC_, HepMCHandle);

    evt = replacer_->produce(*muons, 0, HepMCHandle->GetEvent());
  }
  else if(replacementMode_ == 1){
    evt = replacer_->produce(*muons);
  }
  else
    throw cms::Exception("Configuration") << "Unsupported replacementMode " << replacementMode_ << ", should be 1 or 0" << std::endl;


  if(evt.get() != 0) {
    std::auto_ptr<edm::HepMCProduct> bare_product(new edm::HepMCProduct());  
    bare_product->addHepMCData(evt.release()); // transfer ownership of the HepMC:GenEvent to bare_product

    iEvent.put(bare_product);
  }
}

void MCParticleReplacer::beginRun(edm::Run& iRun,const edm::EventSetup& iSetup)
{
  replacer_->beginRun(iRun, iSetup);
}

void MCParticleReplacer::endRun()
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
