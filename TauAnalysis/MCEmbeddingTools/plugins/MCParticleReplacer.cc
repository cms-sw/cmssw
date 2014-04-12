#include "TauAnalysis/MCEmbeddingTools/plugins/MCParticleReplacer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

// replacementMode =
//	0 - remove Muons from existing HepMCProduct and implant taus (and tau decay products)
//	1 - build new HepMCProduct only with taus (and tau decay products)
MCParticleReplacer::MCParticleReplacer(const edm::ParameterSet& cfg)
  : src_(cfg.getParameter<edm::InputTag>("src")),
    srcHepMC_(cfg.getParameter<edm::InputTag>("hepMcSrc")),
    hepMcMode_(stringToHepMcMode(cfg.getParameter<std::string>("hepMcMode"))),
    replacer_(0),
    evt_(0)
{
  std::string algorithm = cfg.getParameter<std::string>("algorithm");
  edm::ParameterSet cfgAlgorithm = cfg.getParameter<edm::ParameterSet>(algorithm);
  std::string pluginType = cfg.getParameter<std::string>("pluginType");
  replacer_ = ParticleReplacerPluginFactory::get()->create(pluginType, cfgAlgorithm);

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  produces<edm::HepMCProduct>();
  produces<GenFilterInfo>("minVisPtFilter");
  replacer_->declareExtraProducts(this);
}

MCParticleReplacer::~MCParticleReplacer()
{
  delete replacer_;
}

MCParticleReplacer::HepMcMode MCParticleReplacer::stringToHepMcMode(const std::string& name) 
{
  if ( name == "new" ) return kNew;
  else if ( name == "replace" ) return kReplace;
  else throw cms::Exception("Configuration") 
    << "Unsupported hepMcMode " << name << ": should be either 'new' or 'replace' !!\n" << std::endl;
}

void
MCParticleReplacer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, src_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  std::vector<reco::Particle> muons;
  if ( muPlus.isNonnull()  ) muons.push_back(reco::Particle(muPlus->charge(), muPlus->p4(), muPlus->vertex(), -15, 0, true));
  if ( muMinus.isNonnull() ) muons.push_back(reco::Particle(muMinus->charge(), muMinus->p4(), muMinus->vertex(), +15, 0, true));

  if ( muons.size() == 0 ) {
    edm::LogError("MCParticleReplacer") 
      << "No Z->mumu candidates or muons found !!" << std::endl;
    return;
  }
	
  evt_ = &evt;

  std::auto_ptr<HepMC::GenEvent> hepMC;
  if ( hepMcMode_ == kReplace ) {
    edm::Handle<edm::HepMCProduct> HepMCHandle;	 
    evt.getByLabel(srcHepMC_, HepMCHandle);
    hepMC = replacer_->produce(muons, 0, HepMCHandle->GetEvent(), this);
  } else if( hepMcMode_ == kNew ) {
    hepMC = replacer_->produce(muons, 0, 0, this);
  } else
    throw cms::Exception("LogicError") 
      << "Invalid hepMcMode " << hepMcMode_ << " !!" << std::endl;

  if ( hepMC.get() != 0 ) {
    std::auto_ptr<edm::HepMCProduct> bare_product(new edm::HepMCProduct());  
    bare_product->addHepMCData(hepMC.release()); // transfer ownership of the HepMC:GenEvent to bare_product
    
    evt.put(bare_product);
    
    std::auto_ptr<GenFilterInfo> info(new GenFilterInfo(replacer_->tried_, replacer_->passed_));
    evt.put(info, std::string("minVisPtFilter"));
  }
}

void MCParticleReplacer::beginRun(edm::Run& run,const edm::EventSetup& es)
{
  replacer_->beginRun(run, es);
}

void MCParticleReplacer::endRun()
{
  replacer_->endRun();
}

void 
MCParticleReplacer::beginJob()
{
  
  replacer_->beginJob();
}

void
MCParticleReplacer::endJob()
{
  replacer_->endJob();
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MCParticleReplacer);
