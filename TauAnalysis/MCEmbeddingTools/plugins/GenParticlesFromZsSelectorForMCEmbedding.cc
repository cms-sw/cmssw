#include "TauAnalysis/MCEmbeddingTools/plugins/GenParticlesFromZsSelectorForMCEmbedding.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include <TMath.h>

#include <vector>

GenParticlesFromZsSelectorForMCEmbedding::GenParticlesFromZsSelectorForMCEmbedding(const edm::ParameterSet& cfg) 
{
  src_ = cfg.getParameter<edm::InputTag>("src");

  pdgIdsMothers_ = cfg.getParameter<vint>("pdgIdsMothers");
  pdgIdsDaughters_ = cfg.getParameter<vint>("pdgIdsDaughters");

  maxDaughters_ = cfg.getParameter<int>("maxDaughters");
  minDaughters_ = cfg.getParameter<int>("minDaughters");

  std::string before_or_afterFSR_string = cfg.getParameter<std::string>("before_or_afterFSR");
  if      ( before_or_afterFSR_string == "beforeFSR" ) before_or_afterFSR_ = kBeforeFSR;
  else if ( before_or_afterFSR_string == "afterFSR"  ) before_or_afterFSR_ = kAfterFSR;
  else throw cms::Exception("Configuration")
    << " Invalid Configuration Parameter 'before_or_afterFSR' = " << before_or_afterFSR_string << " !!\n";

  verbosity_ = ( cfg.exists("verbosity") ) ? 
    cfg.getParameter<int>("verbosity") : 0;

  produces<reco::GenParticleCollection>("");
}

GenParticlesFromZsSelectorForMCEmbedding::~GenParticlesFromZsSelectorForMCEmbedding() 
{ 
// nothing to be done yet...
}

namespace
{
  void findGenParticles(const reco::GenParticleCollection& genParticles, 
			int pdgIdMother, int pdgIdDaughter, std::vector<const reco::GenParticle*>& genParticlesFromZs)
  {
    for ( reco::GenParticleCollection::const_iterator genParticle = genParticles.begin();
	  genParticle != genParticles.end(); ++genParticle ) {
      if ( TMath::Abs(genParticle->pdgId()) == pdgIdMother ) {
	unsigned numDaughters = genParticle->numberOfDaughters();
	for ( unsigned iDaughter = 0; iDaughter < numDaughters; ++iDaughter ) {
	  const reco::GenParticle* daughter = genParticle->daughterRef(iDaughter).get();
	  if ( TMath::Abs(daughter->pdgId()) == pdgIdDaughter ) genParticlesFromZs.push_back(daughter);
	}
      }
    }
  }

  void findGenParticles(const reco::GenParticleCollection& genParticles, 
			int pdgIdDaughter, int minDaughters, bool requireSubsequentEntries, std::vector<const reco::GenParticle*>& genParticlesFromZs)
  {
    int indexDaughterPlus  = -1;
    int indexDaughterMinus = -1;
    
    unsigned numGenParticles = genParticles.size();
    for ( unsigned iGenParticle = 0; iGenParticle < numGenParticles; ++iGenParticle ) {
      const reco::GenParticle& genParticle = genParticles[iGenParticle];
      
      if ( TMath::Abs(genParticle.pdgId()) == pdgIdDaughter ) {
	if      ( genParticle.charge() > 0 ) indexDaughterPlus  = (int)iGenParticle;
	else if ( genParticle.charge() < 0 ) indexDaughterMinus = (int)iGenParticle;
	
	if ( indexDaughterPlus != -1 && indexDaughterMinus != -1 ) {
	  if ( TMath::Abs(indexDaughterPlus - indexDaughterMinus) == 1 || (!requireSubsequentEntries) ) {
	    genParticlesFromZs.push_back(&genParticles[indexDaughterPlus]);
	    genParticlesFromZs.push_back(&genParticles[indexDaughterMinus]);
	    indexDaughterPlus  = -1;
	    indexDaughterMinus = -1;
	  }
	}
      }

      if ( (int)genParticlesFromZs.size() >= minDaughters ) break;
    }
  }

  void findDaughters(const reco::GenParticle* mother, std::vector<const reco::GenParticle*>& daughters, int status)
  {
    unsigned numDaughters = mother->numberOfDaughters();
    for ( unsigned iDaughter = 0; iDaughter < numDaughters; ++iDaughter ) {
      const reco::GenParticle* daughter = mother->daughterRef(iDaughter).get();
      
      if ( status == -1 || daughter->status() == status ) daughters.push_back(daughter);
      
      findDaughters(daughter, daughters, status);
    }
  }
}  

void GenParticlesFromZsSelectorForMCEmbedding::produce(edm::Event& evt, const edm::EventSetup& es) 
{
  if ( verbosity_ ) {
    std::cout << "<GenParticlesFromZsSelectorForMCEmbedding::produce>:" << std::endl;
    std::cout << " src = " << src_ << std::endl;
  }

  edm::Handle<reco::GenParticleCollection> genParticles;
  evt.getByLabel(src_, genParticles);
  
  std::vector<const reco::GenParticle*> genParticlesFromZs_tmp;

//--- check if HepEVT record contains any Z/gamma* --> l+ l- entries
//
//    NOTE: iteration over mothers in the outer loop
//          gives Z --> l+ l- decays priority over gamma --> e+ e- 
//
  for ( vint::const_iterator pdgIdMother = pdgIdsMothers_.begin();
	pdgIdMother != pdgIdsMothers_.end(); ++pdgIdMother ) {
    for ( vint::const_iterator pdgIdDaughter = pdgIdsDaughters_.begin();
	  pdgIdDaughter != pdgIdsDaughters_.end(); ++pdgIdDaughter ) {
      if ( (int)genParticlesFromZs_tmp.size() < maxDaughters_ || maxDaughters_ == -1 )
	findGenParticles(*genParticles, *pdgIdMother, *pdgIdDaughter, genParticlesFromZs_tmp);
    }
  }
  
//--- check if HepEVT record contains l+ l- entries without Z/gamma* parent
//    
//    NOTE: in order to avoid ambiguities, give preference to l+ l- pairs
//          which are subsequent in HepEVT record
//
  if ( !((int)genParticlesFromZs_tmp.size() >= minDaughters_) ) {
    for ( vint::const_iterator pdgIdDaughter = pdgIdsDaughters_.begin();
	  pdgIdDaughter != pdgIdsDaughters_.end(); ++pdgIdDaughter ) {
      findGenParticles(*genParticles, *pdgIdDaughter, minDaughters_, true, genParticlesFromZs_tmp);
    }
  }
  if ( !((int)genParticlesFromZs_tmp.size() >= minDaughters_) ) {
    for ( vint::const_iterator pdgIdDaughter = pdgIdsDaughters_.begin();
	  pdgIdDaughter != pdgIdsDaughters_.end(); ++pdgIdDaughter ) {
      findGenParticles(*genParticles, *pdgIdDaughter, minDaughters_, false, genParticlesFromZs_tmp);
    }
  }
  
  std::auto_ptr<reco::GenParticleCollection> genParticlesFromZs(new reco::GenParticleCollection());
  
  int idx = 0;
  for ( std::vector<const reco::GenParticle*>::const_iterator genParticleFromZ_beforeFSR = genParticlesFromZs_tmp.begin();
	genParticleFromZ_beforeFSR != genParticlesFromZs_tmp.end(); ++genParticleFromZ_beforeFSR ) {
    if ( before_or_afterFSR_ == kBeforeFSR ) {
      genParticlesFromZs->push_back(**genParticleFromZ_beforeFSR);
    } else if ( before_or_afterFSR_ == kAfterFSR ) {
      std::vector<const reco::GenParticle*> daughters;
      findDaughters(*genParticleFromZ_beforeFSR, daughters, -1);
      const reco::GenParticle* genParticleFromZ_afterFSR = (*genParticleFromZ_beforeFSR);
      for ( std::vector<const reco::GenParticle*>::const_iterator daughter = daughters.begin();
	    daughter != daughters.end(); ++daughter ) {
	if ( (*daughter)->pdgId() == (*genParticleFromZ_beforeFSR)->pdgId() && 
	     (*daughter)->energy() < genParticleFromZ_afterFSR->energy() ) genParticleFromZ_afterFSR = (*daughter);
      }
      if ( verbosity_ ) {
	std::cout << "genParticleFromZ #" << idx << " (beforeFSR): Pt = " << (*genParticleFromZ_beforeFSR)->pt() << "," 
		  << " eta = " << (*genParticleFromZ_beforeFSR)->eta() << ", phi = " << (*genParticleFromZ_beforeFSR)->phi() << std::endl;
	std::cout << "genParticleFromZ #" << idx << " (afterFSR): Pt = " << genParticleFromZ_afterFSR->pt() << "," 
		  << " eta = " << genParticleFromZ_afterFSR->eta() << ", phi = " << genParticleFromZ_afterFSR->phi() << std::endl;
      }
      genParticlesFromZs->push_back(*genParticleFromZ_afterFSR);
    } else assert(0);
    ++idx;
  }
  
  evt.put(genParticlesFromZs);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenParticlesFromZsSelectorForMCEmbedding);
