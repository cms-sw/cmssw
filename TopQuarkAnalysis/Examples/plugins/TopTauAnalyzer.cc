#include "DataFormats/PatCandidates/interface/Tau.h"
#include "TopQuarkAnalysis/Examples/plugins/TopTauAnalyzer.h"


TopTauAnalyzer::TopTauAnalyzer(const edm::ParameterSet& cfg):
  input_(cfg.getParameter<edm::InputTag>("input"))
{
  edm::Service<TFileService> fs;
  
  mult_ = fs->make<TH1F>("mult", "multiplicity (taus)", 30,  0 ,   30);
  en_   = fs->make<TH1F>("en"  , "energy (taus)",       60,  0., 300.);
  pt_   = fs->make<TH1F>("pt"  , "pt (taus}",           60,  0., 300.);
  eta_  = fs->make<TH1F>("eta" , "eta (taus)",          30, -3.,   3.);
  phi_  = fs->make<TH1F>("phi" , "phi (taus)",          40, -4.,   4.);
}

TopTauAnalyzer::~TopTauAnalyzer()
{
}

void
TopTauAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{       
  edm::Handle<std::vector<pat::Tau> > taus;
  evt.getByLabel(input_, taus); 

  // fill histograms

  mult_->Fill( taus->size() );
  for(std::vector<pat::Tau>::const_iterator tau=taus->begin(); tau!=taus->end(); ++tau){
    en_ ->Fill( tau->energy() );
    pt_ ->Fill( tau->pt()     );
    eta_->Fill( tau->eta()    );
    phi_->Fill( tau->phi()    );
  }
}

void TopTauAnalyzer::beginJob()
{
}

void TopTauAnalyzer::endJob()
{
}
  
