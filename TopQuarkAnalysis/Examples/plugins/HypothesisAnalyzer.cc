#include "DataFormats/Candidate/interface/Candidate.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvent.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiEvtPartons.h"
#include "TopQuarkAnalysis/Examples/plugins/HypothesisAnalyzer.h"


HypothesisAnalyzer::HypothesisAnalyzer(const edm::ParameterSet& cfg):
  semiEvt_ (cfg.getParameter<edm::InputTag>("semiEvent")),
  hypoKey_ (cfg.getParameter<edm::InputTag>("hypoKey"  ))
{
}

void
HypothesisAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<TtSemiEvent> semiEvt;
  evt.getByLabel(semiEvt_, semiEvt);

  edm::Handle<int> hypoKeyHandle;
  evt.getByLabel(hypoKey_, hypoKeyHandle);
  TtSemiEvent::HypoKey& hypoKey = (TtSemiEvent::HypoKey&) *hypoKeyHandle;

  if( !semiEvt->isHypoAvailable(hypoKey) ){
    edm::LogWarning ( "NonValidHyp" ) << "Hypothesis not available for this event";
    return;
  }
  if( !semiEvt->isHypoValid(hypoKey) ){
    edm::LogWarning ( "NonValidHyp" ) << "Hypothesis not valid for this event";
    return;
  }
  
  const reco::Candidate* hadTop = semiEvt->hadronicTop(hypoKey);
  const reco::Candidate* hadW   = semiEvt->hadronicW  (hypoKey);
  const reco::Candidate* lepTop = semiEvt->leptonicTop(hypoKey);
  const reco::Candidate* lepW   = semiEvt->leptonicW  (hypoKey);
  
  if(hadTop && hadW && lepTop && lepW){
    hadWPt_    ->Fill( hadW->pt()    );
    hadWMass_  ->Fill( hadW->mass()  );
    hadTopPt_  ->Fill( hadTop->pt()  );
    hadTopMass_->Fill( hadTop->eta() );
    
    lepWPt_    ->Fill( lepW->pt()    );
    lepWMass_  ->Fill( lepW->mass()  );
    lepTopPt_  ->Fill( lepTop->pt()  );
    lepTopMass_->Fill( lepTop->eta() );
  }
}

void 
HypothesisAnalyzer::beginJob(const edm::EventSetup&)
{
  edm::Service<TFileService> fs;
  if( !fs ) throw edm::Exception( edm::errors::Configuration, "TFile Service is not registered in cfg file" );

  hadWPt_     = fs->make<TH1F>("hadWPt",     "p_{t} (W_{had}) [GeV]", 100,  0.,  500.);
  hadWMass_   = fs->make<TH1F>("hadWMass",   "M (W_{had}) [GeV]"    ,  50,  0. , 150.);
  hadTopPt_   = fs->make<TH1F>("hadTopPt",   "p_{t} (t_{had}) [GeV]", 100,  0. , 500.);
  hadTopMass_ = fs->make<TH1F>("hadTopMass", "M (t_{had}) [GeV]",      50, 50. , 250.);

  lepWPt_     = fs->make<TH1F>("lepWPt",     "p_{t} (W_{lep}) [GeV]", 100,  0.,  500.);
  lepWMass_   = fs->make<TH1F>("lepWMass",   "M (W_{lep}) [GeV]"    ,  50,  0. , 150.);
  lepTopPt_   = fs->make<TH1F>("lepTopPt",   "p_{t} (t_{lep}) [GeV]", 100,  0. , 500.);
  lepTopMass_ = fs->make<TH1F>("lepTopMass", "M (t_{lep}) [GeV]",      50, 50. , 250.);
}

void
HypothesisAnalyzer::endJob() 
{
}
