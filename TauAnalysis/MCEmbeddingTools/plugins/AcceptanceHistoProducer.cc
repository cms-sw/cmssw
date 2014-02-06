#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "TauAnalysis/MCEmbeddingTools/plugins/AcceptanceHistoProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

AcceptanceHistoProducer::AcceptanceHistoProducer(const edm::ParameterSet& cfg):
  dqmDir_(cfg.getParameter<std::string>("dqmDir")),
  srcGenParticles_(cfg.getParameter<edm::InputTag>("srcGenParticles")),
  hPtPosPtNeg_(NULL), hEtaPosEtaNeg_(NULL), hPtPosEtaPos_(NULL), hPtNegEtaNeg_(NULL)
{
}

AcceptanceHistoProducer::~AcceptanceHistoProducer()
{
}

void AcceptanceHistoProducer::beginJob()
{
  dbe_ = edm::Service<DQMStore>().operator->();

  dbe_->setCurrentFolder("MCEmbedding/ZmumuAcceptance/" + dqmDir_);
  hPtPosPtNeg_ = dbe_->book2DD("PtPosPtNeg", "Positive muon transverse momentum vs. negative muon transverse momentum;p_{T}^{+};p_{T}^{-}", 500, 0.0, 500.0, 500, 0.0, 500.0);
  hEtaPosEtaNeg_ = dbe_->book2DD("EtaPosEtaNeg", "Positive muon pseudorapdity vs. negative muon pseudorapidity;#eta^{+};#eta^{-}", 500, -2.5, 2.5, 500, -2.5, 2.5);
  hPtPosEtaPos_ = dbe_->book2DD("PtPosEtaPos", "Positive muon transverse momentum vs. positive muon pseudorapidity;p_{T}^{+};#eta^{+}", 500, 0.0, 500.0, 500, -2.5, 2.5);
  hPtNegEtaNeg_ = dbe_->book2DD("PtNegEtaNeg", "Negative muon transverse momentum vs. negative muon pseudorapidity;p_{T}^{-};#eta^{-}", 500, 0.0, 500.0, 500, -2.5, 2.5);

  hPtPosPtNeg_->setResetMe(true);
  hPtPosPtNeg_->setLumiFlag();
  hEtaPosEtaNeg_->setResetMe(true);
  hEtaPosEtaNeg_->setLumiFlag();
  hPtPosEtaPos_->setResetMe(true);
  hPtPosEtaPos_->setLumiFlag();
  hPtNegEtaNeg_->setResetMe(true);
  hPtNegEtaNeg_->setLumiFlag();
}

void AcceptanceHistoProducer::analyze(const edm::Event& evt, const edm::EventSetup& es)
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
    // the decay chain does not end in a muon.
    if(genPosMuon->status() == 1 && genNegMuon->status() == 1)
    {
      hPtPosPtNeg_->Fill(genPosMuon->pt(), genNegMuon->pt());
      hEtaPosEtaNeg_->Fill(genPosMuon->eta(), genNegMuon->eta());
      hPtPosEtaPos_->Fill(genPosMuon->pt(), genPosMuon->eta());
      hPtNegEtaNeg_->Fill(genNegMuon->pt(), genNegMuon->eta());
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(AcceptanceHistoProducer);
