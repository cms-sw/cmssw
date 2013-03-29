#include "TauAnalysis/MCEmbeddingTools/plugins/MixedGenMEtProducer.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"

MixedGenMEtProducer::MixedGenMEtProducer(const edm::ParameterSet& cfg)
{
  srcGenParticles1_   = cfg.getParameter<edm::InputTag>("srcGenParticles1");
  srcGenParticles2_   = cfg.getParameter<edm::InputTag>("srcGenParticles2");

  srcRemovedMuons_ = cfg.getParameter<edm::InputTag>("srcRemovedMuons");

  std::string type_string = cfg.getParameter<std::string>("type");
  if      ( type_string == "pf"   ) type_ = kPF;
  else if ( type_string == "calo" ) type_ = kCalo;
  else throw cms::Exception("Configuration") 
    << "Invalid Configuration Parameter 'type' = " << type_string << " !!\n";

  isMC_ = cfg.getParameter<bool>("isMC");
  int numCollections = 0;
  if ( srcGenParticles1_.label() != "" ) ++numCollections;
  if ( srcGenParticles2_.label() != "" ) ++numCollections;
  if ( numCollections != 2 && isMC_ )
    throw cms::Exception("Configuration") 
      << "Collections 'srcGenParticles1' and 'srcGenParticles2' must both be specified in case Embedding is run on Monte Carlo !!\n";
  if ( numCollections != 1 && !isMC_ )
    throw cms::Exception("Configuration") 
      << "Either collection 'srcGenParticles1' or 'srcGenParticles2' must be specified in case Embedding is run on Data !!\n";
  
  produces<reco::GenMETCollection>();
}

namespace
{
  void sumNeutrinoP4s(const reco::GenParticleCollection& genParticles, bool isCaloMEt, reco::Candidate::LorentzVector& sumNeutrinoP4, double& sumEt)
  {
    int idx = 0;
    for ( reco::GenParticleCollection::const_iterator genParticle = genParticles.begin();
	  genParticle != genParticles.end(); ++genParticle ) {
      if ( genParticle->status() == 1 ) {
	int absPdgId = TMath::Abs(genParticle->pdgId());
	if ( absPdgId == 12 || absPdgId == 14 || absPdgId == 16 ) sumNeutrinoP4 += genParticle->p4();
	else if ( isCaloMEt && absPdgId == 13 ) sumNeutrinoP4 += genParticle->p4();
	else sumEt += genParticle->et();
      }
      ++idx;
    }
  }
}

void MixedGenMEtProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  reco::Candidate::LorentzVector genMEtP4;
  double genSumEt = 0.;

  if ( srcGenParticles1_.label() != "" ) {
    edm::Handle<reco::GenParticleCollection> genParticles1;
    evt.getByLabel(srcGenParticles1_, genParticles1);
    sumNeutrinoP4s(*genParticles1, type_ == kCalo, genMEtP4, genSumEt);
  } 
  if ( srcGenParticles2_.label() != "" ) {
    edm::Handle<reco::GenParticleCollection> genParticles2;
    evt.getByLabel(srcGenParticles2_, genParticles2);
    sumNeutrinoP4s(*genParticles2, type_ == kCalo, genMEtP4, genSumEt);
  }

  if ( isMC_ ) {
    typedef edm::View<reco::Candidate> CandidateView;
    edm::Handle<CandidateView> removedMuons;
    evt.getByLabel(srcRemovedMuons_, removedMuons);
    for ( CandidateView::const_iterator removedMuon = removedMuons->begin();
  	  removedMuon != removedMuons->end(); ++removedMuon ) {
      if ( type_ == kCalo ) genMEtP4 -= removedMuon->p4();
      genSumEt -= removedMuon->et();
    }
  }
  
  std::auto_ptr<reco::GenMETCollection> genMETs(new reco::GenMETCollection());
  SpecificGenMETData genMEtData; // WARNING: not filled
  genMEtData.NeutralEMEtFraction  = 0.;
  genMEtData.NeutralHadEtFraction = 0.;
  genMEtData.ChargedEMEtFraction  = 0.;
  genMEtData.ChargedHadEtFraction = 0.;
  genMEtData.MuonEtFraction       = 0.;
  genMEtData.InvisibleEtFraction  = 0.;
  reco::Candidate::Point vtx(0.0, 0.0, 0.0);  
  genMETs->push_back(reco::GenMET(genMEtData, genSumEt, genMEtP4, vtx));

  evt.put(genMETs);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MixedGenMEtProducer);
