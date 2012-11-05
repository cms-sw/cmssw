#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "TauAnalysis/MCEmbeddingTools/plugins/AcceptanceHistoProducer.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

AcceptanceHistoProducer::AcceptanceHistoProducer(const edm::ParameterSet& cfg):
  srcGenParticles_(cfg.getParameter<edm::InputTag>("srcGenParticles")),
  hPtPosPtNeg_(NULL), hEtaPosEtaNeg_(NULL), hPtPosEtaPos_(NULL), hPtNegEtaNeg_(NULL)
{
  /*produces<TH2D, edm::InLumi>("PtPosPtNeg");
  produces<TH2D, edm::InLumi>("EtaPosEtaNeg");
  produces<TH2D, edm::InLumi>("PtPosEtaPos");
  produces<TH2D, edm::InLumi>("PtNegEtaNeg");*/

  edm::Service<TFileService> fs;
  hPtPosPtNeg_ = fs->make<TH2D>("PtPosPtNeg", "Positive muon transverse momentum vs. negative muon transverse momentum;p_{T}^{+};p_{T}^{-}", 500, 0.0, 500.0, 500, 0.0, 500.0);
  hEtaPosEtaNeg_ = fs->make<TH2D>("EtaPosEtaNeg", "Positive muon pseudorapdity vs. negative muon pseudorapidity;#eta^{+};#eta^{-}", 500, -2.5, 2.5, 500, -2.5, 2.5);
  hPtPosEtaPos_ = fs->make<TH2D>("PtPosEtaPos", "Positive muon transverse momentum vs. positive muon pseudorapidity;p_{T}^{+};#eta^{+}", 500, 0.0, 500.0, 500, -2.5, 2.5);
  hPtNegEtaNeg_ = fs->make<TH2D>("PtNegEtaNeg", "Negative muon transverse momentum vs. negative muon pseudorapidity;p_{T}^{-};#eta^{-}", 500, 0.0, 500.0, 500, -2.5, 2.5);
}

AcceptanceHistoProducer::~AcceptanceHistoProducer()
{
}

void AcceptanceHistoProducer::beginLuminosityBlock(edm::LuminosityBlock& lumi, const edm::EventSetup&)
{
#if 0
  assert(hPtPosPtNeg_ == NULL);
  assert(hEtaPosEtaNeg_ == NULL);
  assert(hPtPosEtaPos_ == NULL);
  assert(hPtNegEtaNeg_ == NULL);

  hPtPosPtNeg_ = new TH2D("PtPosPtNeg", "Positive muon transverse momentum vs. negative muon transverse momentum;p_{T}^{+};p_{T}^{-}", 500, 0.0, 500.0, 500, 0.0, 500.0);
  hEtaPosEtaNeg_ = new TH2D("EtaPosEtaNeg", "Positive muon pseudorapdity vs. negative muon pseudorapidity;#eta^{+};#eta^{-}", 500, -2.5, 2.5, 500, -2.5, 2.5);
  hPtPosEtaPos_ = new TH2D("PtPosEtaPos", "Positive muon transverse momentum vs. positive muon pseudorapidity;p_{T}^{+};#eta^{+}", 500, 0.0, 500.0, 500, -2.5, 2.5);
  hPtNegEtaNeg_ = new TH2D("PtNegEtaNeg", "Negative muon transverse momentum vs. negative muon pseudorapidity;p_{T}^{-};#eta^{-}", 500, 0.0, 500.0, 500, -2.5, 2.5);
#endif
}

void AcceptanceHistoProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  assert(hPtPosPtNeg_ != NULL);
  assert(hEtaPosEtaNeg_ != NULL);
  assert(hPtPosEtaPos_ != NULL);
  assert(hPtNegEtaNeg_ != NULL);

  edm::Handle<reco::GenParticleCollection> genParticles;
  evt.getByLabel(srcGenParticles_, genParticles);

  // Look for the two muons from the matrix element (status=3)
  const reco::GenParticle* genPosMuonME = NULL;
  const reco::GenParticle* genNegMuonME = NULL;
  for(unsigned int i = 0; i < genParticles->size(); ++i)
  {
    const reco::GenParticle& part = (*genParticles)[i];
    if(abs(part.pdgId()) == 13 && part.status() == 3)
    {
      if(part.charge() < 0)
        if(!genNegMuonME)
          genNegMuonME = &part;
      if(part.charge() > 0)
        if(!genPosMuonME)
          genPosMuonME = &part;
    }
  }

  if(genPosMuonME && genNegMuonME)
  {
    const reco::GenParticle* genPosMuon = genPosMuonME;
    const reco::GenParticle* genNegMuon = genNegMuonME;

    // Follow the decay chain to find the stable decay products (status=1)
    while(genPosMuon->status() != 1 && genPosMuon->numberOfDaughters() > 0)
    {
      unsigned int i;
      for(i = 0; i < genPosMuon->numberOfDaughters(); ++i)
      {
        const reco::GenParticle* daughter = dynamic_cast<const reco::GenParticle*>(genPosMuon->daughter(i));
        if(abs(daughter->pdgId()) == 13)
          { genPosMuon = daughter; break; }
      }

      // No more muon daugthers? Maybe mu->e decay...
      if(i == genPosMuon->numberOfDaughters())
        break;
    }

    while(genNegMuon->status() != 1 && genNegMuon->numberOfDaughters() > 0)
    {
      unsigned int i;
      for(i = 0; i < genNegMuon->numberOfDaughters(); ++i)
      {
        const reco::GenParticle* daughter = dynamic_cast<const reco::GenParticle*>(genNegMuon->daughter(i));
        if(abs(daughter->pdgId()) == 13)
          { genNegMuon = daughter; break; }
      }

      // No more muon daugthers? Maybe mu->e decay...
      if(i == genNegMuon->numberOfDaughters())
        break;
    }

    // Loop might have broken earlier if there are no more daugthers, or if
    // the decay chain does not decay in a muon.
    if(genPosMuon->status() == 1 && genNegMuon->status() == 1)
    {
      hPtPosPtNeg_->Fill(genPosMuon->pt(), genNegMuon->pt());
      hEtaPosEtaNeg_->Fill(genPosMuon->eta(), genNegMuon->eta());
      hPtPosEtaPos_->Fill(genPosMuon->pt(), genPosMuon->eta());
      hPtNegEtaNeg_->Fill(genNegMuon->pt(), genNegMuon->eta());
    }
  }
}

void AcceptanceHistoProducer::endLuminosityBlock(edm::LuminosityBlock& lumi, const edm::EventSetup&)
{
#if 0
  assert(hPtPosPtNeg_ != NULL);
  assert(hEtaPosEtaNeg_ != NULL);
  assert(hPtPosEtaPos_ != NULL);
  assert(hPtNegEtaNeg_ != NULL);

  std::auto_ptr<TH2D> hPtPosPtNeg(hPtPosPtNeg_);
  std::auto_ptr<TH2D> hEtaPosEtaNeg(hEtaPosEtaNeg_);
  std::auto_ptr<TH2D> hPtPosEtaPos(hPtPosEtaPos_);
  std::auto_ptr<TH2D> hPtNegEtaNeg(hPtNegEtaNeg_);

  hPtPosPtNeg_ = NULL;
  hEtaPosEtaNeg_ = NULL;
  hPtPosEtaPos_ = NULL;
  hPtNegEtaNeg_ = NULL;

  lumi.put(hPtPosPtNeg, "PtPosPtNeg");
  lumi.put(hEtaPosEtaNeg, "EtaPosEtaNeg");
  lumi.put(hPtPosEtaPos, "PtPosEtaPos");
  lumi.put(hPtNegEtaNeg, "PtNegEtaNeg");
#endif
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AcceptanceHistoProducer);
