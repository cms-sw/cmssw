#include "TauAnalysis/MCEmbeddingTools/plugins/GenParticlesFromZsSelectorForMCEmbedding.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include <TMath.h>

GenParticlesFromZsSelectorForMCEmbedding::GenParticlesFromZsSelectorForMCEmbedding(const edm::ParameterSet& cfg) 
{
  src_ = cfg.getParameter<edm::InputTag>("src");

  pdgIdsMothers_ = cfg.getParameter<vint>("pdgIdsMothers");
  pdgIdsDaughters_ = cfg.getParameter<vint>("pdgIdsDaughters");

  maxDaughters_ = cfg.getParameter<int>("maxDaughters");
  minDaughters_ = cfg.getParameter<int>("minDaughters");

  produces<reco::GenParticleCollection>("");
}

GenParticlesFromZsSelectorForMCEmbedding::~GenParticlesFromZsSelectorForMCEmbedding() 
{ 
// nothing to be done yet...
}

namespace
{
  void findGenParticles(const reco::GenParticleCollection& genParticles, 
			int pdgIdMother, int pdgIdDaughter, reco::GenParticleCollection& genParticlesFromZs)
  {
    for ( reco::GenParticleCollection::const_iterator genParticle = genParticles.begin();
	  genParticle != genParticles.end(); ++genParticle ) {
      if ( TMath::Abs(genParticle->pdgId()) == pdgIdMother ) {
	unsigned numDaughters = genParticle->numberOfDaughters();
	for ( unsigned iDaughter = 0; iDaughter < numDaughters; ++iDaughter ) {
	  const reco::GenParticle* daughter = genParticle->daughterRef(iDaughter).get();
	  if ( TMath::Abs(daughter->pdgId()) == pdgIdDaughter ) genParticlesFromZs.push_back(*daughter);
	}
      }
    }
  }

  void findGenParticles(const reco::GenParticleCollection& genParticles, 
			int pdgIdDaughter, int minDaughters, bool requireSubsequentEntries, reco::GenParticleCollection& genParticlesFromZs)
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
	    genParticlesFromZs.push_back(genParticles[indexDaughterPlus]);
	    genParticlesFromZs.push_back(genParticles[indexDaughterMinus]);
	    indexDaughterPlus  = -1;
	    indexDaughterMinus = -1;
	  }
	}
      }
    
      if ( (int)genParticlesFromZs.size() >= minDaughters ) break;
    }
  }
}  

void GenParticlesFromZsSelectorForMCEmbedding::produce(edm::Event& evt, const edm::EventSetup& es) 
{
  std::auto_ptr<reco::GenParticleCollection> genParticlesFromZs(new reco::GenParticleCollection());

  edm::Handle<reco::GenParticleCollection> genParticles;
  evt.getByLabel(src_, genParticles);
  
//--- check if HepEVT record contains any Z/gamma* --> l+ l- entries
//
//    NOTE: iteration over mothers in the outer loop
//          gives Z --> l+ l- decays priority over gamma --> e+ e- 
//
  for ( vint::const_iterator pdgIdMother = pdgIdsMothers_.begin();
	pdgIdMother != pdgIdsMothers_.end(); ++pdgIdMother ) {
    for ( vint::const_iterator pdgIdDaughter = pdgIdsDaughters_.begin();
	  pdgIdDaughter != pdgIdsDaughters_.end(); ++pdgIdDaughter ) {
      if ( (int)genParticlesFromZs->size() < maxDaughters_ || maxDaughters_ == -1 )
	findGenParticles(*genParticles, *pdgIdMother, *pdgIdDaughter, *genParticlesFromZs);
    }
  }
  
//--- check if HepEVT record contains l+ l- entries without Z/gamma* parent
//    
//    NOTE: in order to avoid ambiguities, give preference to l+ l- pairs
//          which are subsequent in HepEVT record
//
  for ( vint::const_iterator pdgIdDaughter = pdgIdsDaughters_.begin();
	pdgIdDaughter != pdgIdsDaughters_.end(); ++pdgIdDaughter ) {
    findGenParticles(*genParticles, *pdgIdDaughter, minDaughters_, true, *genParticlesFromZs);
  }
  for ( vint::const_iterator pdgIdDaughter = pdgIdsDaughters_.begin();
	pdgIdDaughter != pdgIdsDaughters_.end(); ++pdgIdDaughter ) {
    findGenParticles(*genParticles, *pdgIdDaughter, minDaughters_, false, *genParticlesFromZs);
  }
  
  evt.put(genParticlesFromZs);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenParticlesFromZsSelectorForMCEmbedding);
