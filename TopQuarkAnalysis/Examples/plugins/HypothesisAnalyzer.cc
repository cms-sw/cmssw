#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"

#include "TopQuarkAnalysis/Examples/plugins/HypothesisAnalyzer.h"

HypothesisAnalyzer::HypothesisAnalyzer(const edm::ParameterSet& cfg):
  semiLepEvt_  (cfg.getParameter<edm::InputTag>("semiLepEvent")),
  hypoClassKey_(cfg.getParameter<std::string>("hypoClassKey"))
{
}

void
HypothesisAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // get a handle for the TtSemiLeptonicEvent and a key to the hypothesis
  //////////////////////////////////////////////////////////////////////////////////////////////////
  
  edm::Handle<TtSemiLeptonicEvent> semiLepEvt;
  event.getByLabel(semiLepEvt_, semiLepEvt);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // check if hypothesis is available and valid in this event
  //////////////////////////////////////////////////////////////////////////////////////////////////

  if( !semiLepEvt->isHypoValid(hypoClassKey_) ){
    edm::LogInfo("HypothesisAnalyzer") << "Hypothesis " << hypoClassKey_ << " not valid for this event";
    return;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // get reconstructed top quarks and W bosons from the hypothesis
  //////////////////////////////////////////////////////////////////////////////////////////////////

  const reco::Candidate* hadTop = semiLepEvt->hadronicDecayTop(hypoClassKey_);
  const reco::Candidate* hadW   = semiLepEvt->hadronicDecayW  (hypoClassKey_);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // fill simple histograms with pt, eta and the masses of the reconstructed particles
  //////////////////////////////////////////////////////////////////////////////////////////////////

  if(hadW) {
    hadWPt_  ->Fill( hadW->pt()   );
    hadWEta_ ->Fill( hadW->eta()  );
    hadWMass_->Fill( hadW->mass() );
  }

  if(hadTop) {
    hadTopPt_  ->Fill( hadTop->pt()   );
    hadTopEta_ ->Fill( hadTop->eta()  );
    hadTopMass_->Fill( hadTop->mass() );
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // get genParticles
  //////////////////////////////////////////////////////////////////////////////////////////////////

  const reco::Candidate* genHadTop = semiLepEvt->hadronicDecayTop();
  const reco::Candidate* genHadW   = semiLepEvt->hadronicDecayW();

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // fill pull histograms for pt, eta and the masses of the reconstructed with respect to the generated particles
  //////////////////////////////////////////////////////////////////////////////////////////////////

  if(hadW && genHadW) {
    hadWPullPt_  ->Fill( (hadW->pt() - genHadW->pt()) / genHadW->pt()   );
    hadWPullEta_ ->Fill( (hadW->eta() - genHadW->eta()) / genHadW->eta()  );
    hadWPullMass_->Fill( (hadW->mass() - genHadW->mass()) / genHadW->mass() );
  }

  if(hadTop && genHadTop) {
    hadTopPullPt_  ->Fill( (hadTop->pt() - genHadTop->pt()) / genHadTop->pt()   );
    hadTopPullEta_ ->Fill( (hadTop->eta() - genHadTop->eta()) / genHadTop->eta()  );
    hadTopPullMass_->Fill( (hadTop->mass() - genHadTop->mass()) / genHadTop->mass() );
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // fill histograms with variables describing the quality of the hypotheses
  //////////////////////////////////////////////////////////////////////////////////////////////////

  genMatchDr_->Fill(semiLepEvt->genMatchSumDR());
  mvaDisc_   ->Fill(semiLepEvt->mvaDisc());

  if(hadTop && genHadTop) {

    genMatchDrVsHadTopPullMass_->Fill((hadTop->mass() - genHadTop->mass()) / genHadTop->mass(), semiLepEvt->genMatchSumDR());
    mvaDiscVsHadTopPullMass_   ->Fill((hadTop->mass() - genHadTop->mass()) / genHadTop->mass(), semiLepEvt->mvaDisc());

  }

}

void 
HypothesisAnalyzer::beginJob()
{
  edm::Service<TFileService> fs;
  if( !fs ) throw edm::Exception( edm::errors::Configuration, "TFile Service is not registered in cfg file" );

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // book histograms
  //////////////////////////////////////////////////////////////////////////////////////////////////

  hadWPt_   = fs->make<TH1F>("hadWPt"  , "p_{t} (W_{had}) [GeV]", 25,  0., 500.);
  hadWEta_  = fs->make<TH1F>("hadWEta" , "#eta (W_{had})"       , 20, -4.,   4.);
  hadWMass_ = fs->make<TH1F>("hadWMass", "M (W_{had}) [GeV]"    , 25,  0., 200.);

  hadTopPt_   = fs->make<TH1F>("hadTopPt"  , "p_{t} (t_{had}) [GeV]", 25, 0. , 500.);
  hadTopEta_  = fs->make<TH1F>("hadTopEta" , "#eta (t_{had})"       , 20, -4.,   4.);
  hadTopMass_ = fs->make<TH1F>("hadTopMass", "M (t_{had}) [GeV]"    , 40, 0. , 400.);

  hadWPullPt_   = fs->make<TH1F>("hadWPullPt"  , "(p_{t,rec}-p_{t,gen})/p_{t,gen} (W_{had})"   , 40, -1., 1.);
  hadWPullEta_  = fs->make<TH1F>("hadWPullEta" , "(#eta_{rec}-#eta_{gen})/#eta_{gen} (W_{had})", 40, -1., 1.);
  hadWPullMass_ = fs->make<TH1F>("hadWPullMass", "(M_{rec}-M_{gen})/M_{gen} (W_{had})"         , 40, -1., 1.);

  hadTopPullPt_   = fs->make<TH1F>("hadTopPullPt"  , "(p_{t,rec}-p_{t,gen})/p_{t,gen} (t_{had})"   , 40, -1., 1.);
  hadTopPullEta_  = fs->make<TH1F>("hadTopPullEta" , "(#eta_{rec}-#eta_{gen})/#eta_{gen} (t_{had})", 40, -1., 1.);
  hadTopPullMass_ = fs->make<TH1F>("hadTopPullMass", "(M_{rec}-M_{gen})/M_{gen} (t_{had})"         , 40, -1., 1.);

  genMatchDr_ = fs->make<TH1F>("genMatchDr", "GenMatch #Sigma #Delta R", 40, 0., 4.);
  mvaDisc_    = fs->make<TH1F>("mvaDisc"   , "MVA discriminator"       , 20, 0., 1.);

  genMatchDrVsHadTopPullMass_ = fs->make<TH2F>("genMatchDrVsHadTopPullMass",
					       "GenMatch #Sigma #Delta R vs. (M_{rec}-M_{gen})/M_{gen} (t_{had}))",
					       40, -1., 1., 40, 0., 4.);
  mvaDiscVsHadTopPullMass_    = fs->make<TH2F>("mvaDiscVsHadTopPullMass",
					       "MVA discriminator vs. (M_{rec}-M_{gen})/M_{gen} (t_{had}))",
					       40, -1., 1., 20, 0., 1.);
}

void
HypothesisAnalyzer::endJob() 
{
}
