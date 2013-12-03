#include "TauAnalysis/MCEmbeddingTools/plugins/GenMuonRadiationFilter.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <TMath.h>

GenMuonRadiationFilter::GenMuonRadiationFilter(const edm::ParameterSet& cfg)
  : numWarnings_(0),
    maxWarnings_(3)
{
  srcGenParticles_ = cfg.getParameter<edm::InputTag>("srcGenParticles");

  minPtLow_        = cfg.getParameter<double>("minPtLow");
  dRlowPt_         = cfg.getParameter<double>("dRlowPt");

  minPtHigh_       = cfg.getParameter<double>("minPtHigh");
  dRhighPt_        = cfg.getParameter<double>("dRhighPt");  

  invert_          = cfg.getParameter<bool>("invert");
  filter_          = cfg.getParameter<bool>("filter");
  if ( !filter_ ) {
    produces<bool>();
  }

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

namespace
{
  const reco::GenParticle* findGenParticle(const reco::GenParticleCollection& genParticles, int pdgId, int status)
  {
    for ( reco::GenParticleCollection::const_iterator genParticle = genParticles.begin();
	  genParticle != genParticles.end(); ++genParticle ) {
      if ( (status == -1 || genParticle->status() == status) && genParticle->pdgId() == pdgId ) return &(*genParticle);	
    }
    return 0;
  }

  const reco::GenParticle* findGenParticle(const std::vector<const reco::GenParticle*>& genParticles, int pdgId, int status)
  {
    for ( std::vector<const reco::GenParticle*>::const_iterator genParticle = genParticles.begin();
	  genParticle != genParticles.end(); ++genParticle ) {
      if ( (status == -1 || (*genParticle)->status() == status) && (*genParticle)->pdgId() == pdgId ) return (*genParticle);	
    }
    return 0;
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

  void print(const reco::GenParticle* genMuon_beforeFSR, const reco::GenParticle* genPhoton, double dR, const reco::GenParticle* genMuon_afterFSR)
  {
    if ( genMuon_beforeFSR ) 
      std::cout << "muon (before FSR): Pt = " << genMuon_beforeFSR->pt() << ", eta = " << genMuon_beforeFSR->eta() << ", phi = " << genMuon_beforeFSR->phi() 
		<< " (charge = " << genMuon_beforeFSR->charge() << ")" << std::endl;
    if ( genPhoton )
      std::cout << "photon: Pt = " << genPhoton->pt() << ", eta = " << genPhoton->eta() << ", phi = " << genPhoton->phi() << " (dR = " << dR << ")" << std::endl;
    if ( genMuon_afterFSR ) 
      std::cout << "muon (after FSR): Pt = " << genMuon_afterFSR->pt() << ", eta = " << genMuon_afterFSR->eta() << ", phi = " << genMuon_afterFSR->phi() 
		<< " (charge = " << genMuon_afterFSR->charge() << ")" << std::endl;
  }
}

bool GenMuonRadiationFilter::filter(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) std::cout << "<GenMuonRadiationFilter::filter>:" << std::endl;

  edm::Handle<reco::GenParticleCollection> genParticles;
  evt.getByLabel(srcGenParticles_, genParticles);

  std::vector<const reco::GenParticle*> genMothers;
  const reco::GenParticle* genZ                 = findGenParticle(*genParticles, 23, -1);
  const reco::GenParticle* genMuPlus_beforeFSR  = 0;
  const reco::GenParticle* genMuMinus_beforeFSR = 0;
  if ( genZ ) { 
    genMothers.push_back(genZ);
    std::vector<const reco::GenParticle*> genDaughters;
    findDaughters(genZ, genDaughters, -1);
    genMuPlus_beforeFSR  = findGenParticle(genDaughters, -13, -1);
    genMuMinus_beforeFSR = findGenParticle(genDaughters, +13, -1);
  } else {
    genMuPlus_beforeFSR  = findGenParticle(*genParticles, -13, -1);
    genMuMinus_beforeFSR = findGenParticle(*genParticles, +13, -1);
    if ( genMuPlus_beforeFSR && genMuMinus_beforeFSR ) {
      genMothers.push_back(genMuPlus_beforeFSR);
      genMothers.push_back(genMuMinus_beforeFSR);
    }
  }

  if ( !(genMuPlus_beforeFSR && genMuMinus_beforeFSR) ) {
    if ( numWarnings_ < maxWarnings_ ) {
      edm::LogWarning ("<GenMuonRadiationFilter>")
	<< "Failed to find generator level muon pair --> skipping !!" << std::endl;
      ++numWarnings_;
    }
    return false;
  }

  std::vector<const reco::GenParticle*> genDaughters;
  for ( std::vector<const reco::GenParticle*>::const_iterator genMother = genMothers.begin();
	genMother != genMothers.end(); ++genMother ) {
    findDaughters(*genMother, genDaughters, -1);
  }

  bool isMuonRadiation = false;

  for ( std::vector<const reco::GenParticle*>::const_iterator genDaughter = genDaughters.begin();
	genDaughter != genDaughters.end(); ++genDaughter ) {
    if ( (*genDaughter)->pdgId() == 22 ) {
      double dRmuPlus = deltaR((*genDaughter)->p4(), genMuPlus_beforeFSR->p4());
      if ( ((*genDaughter)->pt() > minPtLow_  && dRmuPlus < dRlowPt_ ) ||
	   ((*genDaughter)->pt() > minPtHigh_ && dRmuPlus < dRhighPt_) ) {
	if ( verbosity_ ) {
	  const reco::GenParticle* genMuPlus_afterFSR = findGenParticle(genDaughters, -13, 1);
	  print(genMuPlus_beforeFSR, *genDaughter, dRmuPlus, genMuPlus_afterFSR);
	}
	isMuonRadiation = true;
      }
      double dRmuMinus = deltaR((*genDaughter)->p4(), genMuMinus_beforeFSR->p4());
      if ( ((*genDaughter)->pt() > minPtLow_  && dRmuMinus < dRlowPt_ ) ||
	   ((*genDaughter)->pt() > minPtHigh_ && dRmuMinus < dRhighPt_) ) {
	if ( verbosity_ ) {
	  const reco::GenParticle* genMuMinus_afterFSR = findGenParticle(genDaughters, +13, 1);
	  print(genMuMinus_beforeFSR, *genDaughter, dRmuMinus, genMuMinus_afterFSR);
	}
	isMuonRadiation = true;
      }
    }
  }

  if ( filter_ ) {
    if ( invert_ != isMuonRadiation ) return false; // reject events with muon -> muon + photon radiation
    else return true;
  } else {
    std::auto_ptr<bool> filter_result(new bool(invert_ != !isMuonRadiation));
    evt.put(filter_result);
    return true;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenMuonRadiationFilter);
