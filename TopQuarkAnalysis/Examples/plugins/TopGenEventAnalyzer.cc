#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/Examples/plugins/TopGenEventAnalyzer.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"
 
TopGenEventAnalyzer::TopGenEventAnalyzer(const edm::ParameterSet& cfg):
  inputGenEvent_(cfg.getParameter<edm::InputTag>("genEvent"))
{ 
  edm::Service<TFileService> fs;
  nLep_      = fs->make<TH1F>("nLep",      "N(Lepton)",     5,   0.,   5.);
  topPt_     = fs->make<TH1F>("topPt",     "pt (top)",    100,   0., 500.);
  topEta_    = fs->make<TH1F>("topEta",    "eta(top)",     40,  -5.,   5.);
  topPhi_    = fs->make<TH1F>("topPhi",    "phi(top)",     60, -3.5,  3.5);
  topMass_   = fs->make<TH1F>("topMass",   "mass(top)",   150, 100., 250.);
  topBarPt_  = fs->make<TH1F>("topBarPt",  "pt (topBar)", 100,   0., 500.);
  topBarEta_ = fs->make<TH1F>("topBarEta", "eta(topBar)",  40,  -5.,   5.);
  topBarPhi_ = fs->make<TH1F>("topBarPhi", "phi(topBar)",  60, -3.5,  3.5);
  topBarMass_= fs->make<TH1F>("topBarMass","mass(top)",   150, 100., 250.);
  ttbarPt_   = fs->make<TH1F>("ttbarPt",   "pt (ttbar)",  100,   0., 500.);
  ttbarEta_  = fs->make<TH1F>("ttbarEta",  "eta(ttbar)",   40,  -5.,   5.);
  ttbarPhi_  = fs->make<TH1F>("ttbarPhi",  "phi(ttbar)",   60, -3.5,  3.5);
  ttbarMass_ = fs->make<TH1F>("ttbarMass", "mass(ttbar)", 150, 100., 250.);
}

TopGenEventAnalyzer::~TopGenEventAnalyzer()
{
}

void
TopGenEventAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<TtGenEvent> genEvent;
  evt.getByLabel(inputGenEvent_, genEvent);

  if(genEvent->isFullLeptonic(true)){
    genEvent->dumpEventContent();
    edm::LogVerbatim log("TopGenEventAnalyzer::selection");
    log << "!!! - is full-leptonic - !!! \n";
  }

  // fill BR's
  nLep_  ->Fill(genEvent->numberOfLeptons());

  //fill top kinematic
  topPt_     ->Fill(genEvent->top   ()->pt  ());
  topEta_    ->Fill(genEvent->top   ()->eta ());
  topPhi_    ->Fill(genEvent->top   ()->phi ());
  topMass_   ->Fill(genEvent->top   ()->mass());
  topBarPt_  ->Fill(genEvent->topBar()->pt  ());
  topBarEta_ ->Fill(genEvent->topBar()->eta ());
  topBarPhi_ ->Fill(genEvent->topBar()->phi ());
  topBarMass_->Fill(genEvent->topBar()->mass());

  //fill ttbar kinematics
  reco::Particle::LorentzVector p4 = genEvent->top()->p4()+genEvent->topBar()->p4();
  ttbarPt_  ->Fill(p4.pt  ());
  ttbarEta_ ->Fill(p4.eta ());
  ttbarPhi_ ->Fill(p4.phi ());
  ttbarMass_->Fill(p4.mass());
}

void TopGenEventAnalyzer::beginJob(const edm::EventSetup&)
{  
} 

void TopGenEventAnalyzer::endJob()
{
}
