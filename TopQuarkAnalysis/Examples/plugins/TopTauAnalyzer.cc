#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Flags.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "TopQuarkAnalysis/Examples/plugins/TopTauAnalyzer.h"


TopTauAnalyzer::TopTauAnalyzer(const edm::ParameterSet& cfg):
  taus_(cfg.getParameter<edm::InputTag>("input"))
{
  edm::Service<TFileService> fs;
  
  NrTau_ = fs->make<TH1I>("NrTau",  "Num_{Taus}",    10,  0 , 10 );
  ptTau_ = fs->make<TH1F>("ptTau",  "pt_{Taus}",    100,  0.,300.);
  enTau_ = fs->make<TH1F>("enTau",  "energy_{Taus}",100,  0.,300.);
  etaTau_= fs->make<TH1F>("etaTau", "eta_{Taus}",   100, -3.,  3.);
  phiTau_= fs->make<TH1F>("phiTau", "phi_{Taus}",   100, -4.,  4.);
}

TopTauAnalyzer::~TopTauAnalyzer()
{
}

void
TopTauAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{       
  edm::Handle<std::vector<pat::Tau> > taus;
  evt.getByLabel(taus_, taus); 

  NrTau_->Fill( taus->size() );
  for( std::vector<pat::Tau>::const_iterator tau=taus->begin();
       tau!=taus->end(); ++tau){
    // --------------------------------------------------
    // fill basic tau kinematics 
    // --------------------------------------------------
    ptTau_ ->Fill( tau->pt()    );
    enTau_ ->Fill( tau->energy());
    etaTau_->Fill( tau->eta()   );
    phiTau_->Fill( tau->phi()   );
  }
}

void TopTauAnalyzer::beginJob(const edm::EventSetup&)
{
}

void TopTauAnalyzer::endJob()
{
}
  
