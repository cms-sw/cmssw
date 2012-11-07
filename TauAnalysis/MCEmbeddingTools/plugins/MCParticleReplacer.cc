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

// replacementMode =
//	0 - remove Muons from existing HepMCProduct and implant taus (and tau decay products)
//	1 - build new HepMCProduct only with taus (and tau decay products)
MCParticleReplacer::MCParticleReplacer(const edm::ParameterSet& cfg)
  : src_(cfg.getParameter<edm::InputTag>("src")),
    srcHepMC_(cfg.getParameter<edm::InputTag>("hepMcSrc")),
    hepMcMode_(stringToHepMcMode(cfg.getParameter<std::string>("hepMcMode"))),
    replacer_()
{
  std::string algorithm = cfg.getParameter<std::string>("algorithm");
  edm::ParameterSet cfgAlgorithm = cfg.getParameter<edm::ParameterSet>(algorithm);
  std::string pluginType = cfg.getParameter<std::string>("pluginType");
  replacer_ = ParticleReplacerPluginFactory::get()->create(pluginType, cfgAlgorithm);

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  produces<edm::HepMCProduct>();
  produces<GenFilterInfo>("minVisPtFilter");
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
  std::vector<reco::Particle> muons;
  //   
  // NOTE: the following logic of finding "the" muon pair needs to be kept in synch
  //       between ZmumuPFEmbedder and MCParticleReplacer modules
  //
  edm::Handle<reco::CompositeCandidateCollection> combCandidatesHandle;
  if ( evt.getByLabel(src_, combCandidatesHandle) ) {
    if ( verbosity_ ) std::cout << "<MCParticleReplacer::produce>: #Zs = " << combCandidatesHandle->size() << std::endl;
    if ( combCandidatesHandle->size() >= 1 ) {
      const reco::CompositeCandidate& combCandidate = combCandidatesHandle->at(0); // TF: use only the first combined candidate
      for ( size_t idx = 0; idx < combCandidate.numberOfDaughters(); ++idx ) {
	int charge = combCandidate.daughter(idx)->charge();
	reco::Particle::LorentzVector p4 = combCandidate.daughter(idx)->p4();
	reco::Particle::Point vtx = combCandidate.daughter(idx)->vertex();
	muons.push_back(reco::Particle(charge, p4, vtx, -15*charge, 0, true));
      }
    }
  } else {
    typedef edm::View<reco::Candidate> CandidateView;
    edm::Handle<CandidateView> candsHandle;
    if ( evt.getByLabel(src_, candsHandle) ) {
      if ( verbosity_ ) std::cout << "<MCParticleReplacer::produce>: #muons = " << candsHandle->size() << std::endl;
      for ( size_t idx = 0; idx < candsHandle->size(); ++idx ) {
	int charge = candsHandle->at(idx).charge();
	reco::Particle::LorentzVector p4 = candsHandle->at(idx).p4();
	reco::Particle::Point vtx = candsHandle->at(idx).vertex();
	muons.push_back(reco::Particle(charge, p4, vtx, -15*charge, 0, true));
      }
    } else {
      throw cms::Exception("Configuration") 
	<< "Invalid input collection 'src' = " << src_ << " !!\n";
    }
  }

  if ( muons.size() == 0 ) {
    edm::LogError("MCParticleReplacer") 
      << "No Z -> mumu candidates or muons found !!" << std::endl;
    return;
  }
	
  std::auto_ptr<HepMC::GenEvent> hepMC;
  if ( hepMcMode_ == kReplace ) {
    edm::Handle<edm::HepMCProduct> HepMCHandle;	 
    evt.getByLabel(srcHepMC_, HepMCHandle);
    hepMC = replacer_->produce(muons, 0, HepMCHandle->GetEvent());
  } else if( hepMcMode_ == kNew ) {
    hepMC = replacer_->produce(muons);
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
