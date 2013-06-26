#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/Examples/plugins/TopGenEventAnalyzer.h"
 
TopGenEventAnalyzer::TopGenEventAnalyzer(const edm::ParameterSet& cfg):
  inputGenEvent_(cfg.getParameter<edm::InputTag>("genEvent"))
{ 
  edm::Service<TFileService> fs;
  nLep_      = fs->make<TH1F>("nLep",      "N(Lepton)",     5,   0.,   5.);
  topPt_     = fs->make<TH1F>("topPt",     "pt (top)",    100,   0., 500.);
  topEta_    = fs->make<TH1F>("topEta",    "eta(top)",     40,  -5.,   5.);
  topPhi_    = fs->make<TH1F>("topPhi",    "phi(top)",     60, -3.5,  3.5);
  topBarPt_  = fs->make<TH1F>("topBarPt",  "pt (topBar)", 100,   0., 500.);
  topBarEta_ = fs->make<TH1F>("topBarEta", "eta(topBar)",  40,  -5.,   5.);
  topBarPhi_ = fs->make<TH1F>("topBarPhi", "phi(topBar)",  60, -3.5,  3.5);
  ttbarPt_   = fs->make<TH1F>("ttbarPt",   "pt (ttbar)",  100,   0., 500.);
  ttbarEta_  = fs->make<TH1F>("ttbarEta",  "eta(ttbar)",   40,  -5.,   5.);
  ttbarPhi_  = fs->make<TH1F>("ttbarPhi",  "phi(ttbar)",   60, -3.5,  3.5);
  prodChan_  = fs->make<TH1F>("prodChan",  "production mode", 3, 0, 3);
  prodChan_->GetXaxis()->SetBinLabel(1, "gg"   );
  prodChan_->GetXaxis()->SetBinLabel(2, "qqbar");
  prodChan_->GetXaxis()->SetBinLabel(3, "other");
}

TopGenEventAnalyzer::~TopGenEventAnalyzer()
{
}

void
TopGenEventAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<TtGenEvent> genEvent;
  evt.getByLabel(inputGenEvent_, genEvent);

  if(!genEvent->isTtBar())
    return;

  if(genEvent->fromGluonFusion())
    prodChan_->Fill("gg", 1);
  else if(genEvent->fromQuarkAnnihilation())
    prodChan_->Fill("qqbar", 1);
  else
    prodChan_->Fill("other", 1);

  // fill BR's
  nLep_  ->Fill(genEvent->numberOfLeptons());

  //fill top kinematic
  topPt_    ->Fill(genEvent->top   ()->pt ());
  topEta_   ->Fill(genEvent->top   ()->eta());
  topPhi_   ->Fill(genEvent->top   ()->phi());
  topBarPt_ ->Fill(genEvent->topBar()->pt ());
  topBarEta_->Fill(genEvent->topBar()->eta());
  topBarPhi_->Fill(genEvent->topBar()->phi());

  //fill ttbar kinematics
  ttbarPt_ ->Fill(genEvent->topPair()->pt() );
  ttbarEta_->Fill(genEvent->topPair()->eta());
  ttbarPhi_->Fill(genEvent->topPair()->phi());
}

void TopGenEventAnalyzer::beginJob()
{  
} 

void TopGenEventAnalyzer::endJob()
{
}
