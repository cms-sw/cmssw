#include "TauAnalysis/MCEmbeddingTools/plugins/MixedGenMEtProducer.h"

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

  srcGenRemovedMuons_ = cfg.getParameter<edm::InputTag>("srcGenRemovedMuons");

  produces<reco::GenMETCollection>();
}

namespace
{
  void sumNeutrinoP4s(const reco::GenParticleCollection& genParticles, reco::Candidate::LorentzVector& sumNeutrinoP4, double& sumEt)
  {
    for ( reco::GenParticleCollection::const_iterator genParticle = genParticles.begin();
	  genParticle != genParticles.end(); ++genParticle ) {
      if ( genParticle->status() == 1 ) {
	int absPdgId = TMath::Abs(genParticle->pdgId());
	if ( absPdgId == 12 || absPdgId == 14 || absPdgId == 16 ) sumNeutrinoP4 += genParticle->p4();
	else sumEt += genParticle->et();
      }
    }
  }
}

void MixedGenMEtProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  edm::Handle<reco::GenParticleCollection> genParticles1;
  evt.getByLabel(srcGenParticles1_, genParticles1);

  edm::Handle<reco::GenParticleCollection> genParticles2;
  evt.getByLabel(srcGenParticles2_, genParticles2);

  reco::Candidate::LorentzVector genMEtP4;
  double genSumEt = 0.;
  sumNeutrinoP4s(*genParticles1, genMEtP4, genSumEt);
  sumNeutrinoP4s(*genParticles2, genMEtP4, genSumEt);

  typedef edm::View<reco::Candidate> CandidateView;
  edm::Handle<CandidateView> genRemovedMuons;
  evt.getByLabel(srcGenRemovedMuons_, genRemovedMuons);

  for ( CandidateView::const_iterator genRemovedMuon = genRemovedMuons->begin();
	genRemovedMuon != genRemovedMuons->end(); ++genRemovedMuon ) {
    genSumEt -= genRemovedMuon->et();
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
