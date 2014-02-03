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
  // get reconstructed top quarks, W bosons, the top pair and the neutrino from the hypothesis
  //////////////////////////////////////////////////////////////////////////////////////////////////

  const reco::Candidate* topPair  = semiLepEvt->topPair(hypoClassKey_);
  const reco::Candidate* lepTop   = semiLepEvt->leptonicDecayTop(hypoClassKey_);
  const reco::Candidate* lepW     = semiLepEvt->leptonicDecayW(hypoClassKey_);
  const reco::Candidate* hadTop   = semiLepEvt->hadronicDecayTop(hypoClassKey_);
  const reco::Candidate* hadW     = semiLepEvt->hadronicDecayW(hypoClassKey_);
  const reco::Candidate* neutrino = semiLepEvt->singleNeutrino(hypoClassKey_);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // fill simple histograms with kinematic variables of the reconstructed particles
  //////////////////////////////////////////////////////////////////////////////////////////////////

  if(topPair)
    topPairMass_->Fill( topPair->mass() );
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
  if(lepW) {
    lepWPt_  ->Fill( lepW->pt()   );
    lepWEta_ ->Fill( lepW->eta()  );
    lepWMass_->Fill( lepW->mass() );
  }
  if(lepTop) {
    lepTopPt_  ->Fill( lepTop->pt()   );
    lepTopEta_ ->Fill( lepTop->eta()  );
    lepTopMass_->Fill( lepTop->mass() );
  }
  if(neutrino)
    neutrinoEta_->Fill( neutrino->eta() );

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // get corresponding genParticles
  //////////////////////////////////////////////////////////////////////////////////////////////////

  const math::XYZTLorentzVector* genTopPair = semiLepEvt->topPair();
  const reco::Candidate* genHadTop   = semiLepEvt->hadronicDecayTop();
  const reco::Candidate* genHadW     = semiLepEvt->hadronicDecayW();
  const reco::Candidate* genLepTop   = semiLepEvt->leptonicDecayTop();
  const reco::Candidate* genLepW     = semiLepEvt->leptonicDecayW();
  const reco::Candidate* genNeutrino = semiLepEvt->singleNeutrino();

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // fill pull histograms of kinematic variables with respect to the generated particles
  //////////////////////////////////////////////////////////////////////////////////////////////////

  if(topPair && genTopPair)
    topPairPullMass_->Fill( (topPair->mass()-genTopPair->mass())/ genTopPair->mass() );
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
  if(lepW && genLepW) {
    lepWPullPt_  ->Fill( (lepW->pt() - genLepW->pt()) / genLepW->pt()   );
    lepWPullEta_ ->Fill( (lepW->eta() - genLepW->eta()) / genLepW->eta()  );
    lepWPullMass_->Fill( (lepW->mass() - genLepW->mass()) / genLepW->mass() );
  }

  if(lepTop && genLepTop) {
    lepTopPullPt_  ->Fill( (lepTop->pt() - genLepTop->pt()) / genLepTop->pt()   );
    lepTopPullEta_ ->Fill( (lepTop->eta() - genLepTop->eta()) / genLepTop->eta()  );
    lepTopPullMass_->Fill( (lepTop->mass() - genLepTop->mass()) / genLepTop->mass() );
  }
  if(neutrino && genNeutrino)
    neutrinoPullEta_->Fill( (neutrino->eta()-genNeutrino->eta()) / genNeutrino->eta() );

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // fill histograms with variables describing the quality of the hypotheses
  //////////////////////////////////////////////////////////////////////////////////////////////////

  genMatchDr_->Fill(semiLepEvt->genMatchSumDR());
  kinFitProb_->Fill(semiLepEvt->fitProb());

  if(hadTop && genHadTop) {
    genMatchDrVsHadTopPullMass_->Fill((hadTop->mass() - genHadTop->mass()) / genHadTop->mass(), semiLepEvt->genMatchSumDR());
    kinFitProbVsHadTopPullMass_->Fill((hadTop->mass() - genHadTop->mass()) / genHadTop->mass(), semiLepEvt->fitProb());
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

  neutrinoEta_ = fs->make<TH1F>("neutrinoEta", "#eta (neutrino)", 21, -4., 4.);
  neutrinoPullEta_ = fs->make<TH1F>("neutrinoPullEta", "(#eta_{rec}-#eta_{gen})/#eta_{gen} (neutrino)", 40,  -1., 1.);

  hadWPt_   = fs->make<TH1F>("hadWPt"  , "p_{T} (W_{had}) [GeV]", 25,  0., 500.);
  hadWEta_  = fs->make<TH1F>("hadWEta" , "#eta (W_{had})"       , 21, -4.,   4.);
  hadWMass_ = fs->make<TH1F>("hadWMass", "M (W_{had}) [GeV]"    , 25,  0., 200.);

  hadTopPt_   = fs->make<TH1F>("hadTopPt"  , "p_{T} (t_{had}) [GeV]", 25, 0. , 500.);
  hadTopEta_  = fs->make<TH1F>("hadTopEta" , "#eta (t_{had})"       , 21, -4.,   4.);
  hadTopMass_ = fs->make<TH1F>("hadTopMass", "M (t_{had}) [GeV]"    , 40, 0. , 400.);

  lepWPt_   = fs->make<TH1F>("lepWPt"  , "p_{t} (W_{lep}) [GeV]", 25,  0., 500.);
  lepWEta_  = fs->make<TH1F>("lepWEta" , "#eta (W_{lep})"       , 21, -4.,   4.);
  lepWMass_ = fs->make<TH1F>("lepWMass", "M (W_{lep}) [GeV]"    , 25,  0., 200.);

  lepTopPt_   = fs->make<TH1F>("lepTopPt"  , "p_{T} (t_{lep}) [GeV]", 25, 0. , 500.);
  lepTopEta_  = fs->make<TH1F>("lepTopEta" , "#eta (t_{lep})"       , 21, -4.,   4.);
  lepTopMass_ = fs->make<TH1F>("lepTopMass", "M (t_{lep}) [GeV]"    , 40, 0. , 400.);

  hadWPullPt_   = fs->make<TH1F>("hadWPullPt"  , "(p_{T,rec}-p_{T,gen})/p_{T,gen} (W_{had})"   , 40, -1., 1.);
  hadWPullEta_  = fs->make<TH1F>("hadWPullEta" , "(#eta_{rec}-#eta_{gen})/#eta_{gen} (W_{had})", 40, -1., 1.);
  hadWPullMass_ = fs->make<TH1F>("hadWPullMass", "(M_{rec}-M_{gen})/M_{gen} (W_{had})"         , 40, -1., 1.);

  hadTopPullPt_   = fs->make<TH1F>("hadTopPullPt"  , "(p_{T,rec}-p_{T,gen})/p_{T,gen} (t_{had})"   , 40, -1., 1.);
  hadTopPullEta_  = fs->make<TH1F>("hadTopPullEta" , "(#eta_{rec}-#eta_{gen})/#eta_{gen} (t_{had})", 40, -1., 1.);
  hadTopPullMass_ = fs->make<TH1F>("hadTopPullMass", "(M_{rec}-M_{gen})/M_{gen} (t_{had})"         , 40, -1., 1.);

  lepWPullPt_   = fs->make<TH1F>("lepWPullPt"  , "(p_{T,rec}-p_{T,gen})/p_{T,gen} (W_{lep})"   , 40, -1., 1.);
  lepWPullEta_  = fs->make<TH1F>("lepWPullEta" , "(#eta_{rec}-#eta_{gen})/#eta_{gen} (W_{lep})", 40, -1., 1.);
  lepWPullMass_ = fs->make<TH1F>("lepWPullMass", "(M_{rec}-M_{gen})/M_{gen} (W_{lep})"         , 40, -1., 1.);

  lepTopPullPt_   = fs->make<TH1F>("lepTopPullPt"  , "(p_{T,rec}-p_{T,gen})/p_{T,gen} (t_{lep})"   , 40, -1., 1.);
  lepTopPullEta_  = fs->make<TH1F>("lepTopPullEta" , "(#eta_{rec}-#eta_{gen})/#eta_{gen} (t_{lep})", 40, -1., 1.);
  lepTopPullMass_ = fs->make<TH1F>("lepTopPullMass", "(M_{rec}-M_{gen})/M_{gen} (t_{lep})"         , 40, -1., 1.);

  topPairMass_ = fs->make<TH1F>("topPairMass", "M (t#bar{t})", 36, 340., 940.);
  topPairPullMass_ = fs->make<TH1F>("topPairPullMass", "(M_{rec}-M_{gen})/M_{gen} (t#bar{t})", 40,  -1., 1.);

  genMatchDr_ = fs->make<TH1F>("genMatchDr", "GenMatch #Sigma#DeltaR", 40, 0., 4.);
  kinFitProb_ = fs->make<TH1F>("kinFitProb", "KinFit probability"      , 50, 0., 1.);

  genMatchDrVsHadTopPullMass_ = fs->make<TH2F>("genMatchDrVsHadTopPullMass",
					       "GenMatch #Sigma #Delta R vs. (M_{rec}-M_{gen})/M_{gen} (t_{had}))",
					       40, -1., 1., 40, 0., 4.);
  kinFitProbVsHadTopPullMass_ = fs->make<TH2F>("kinFitProbVsHadTopPullMass",
					       "KinFit probability vs. (M_{rec}-M_{gen})/M_{gen} (t_{had}))",
					       40, -1., 1., 20, 0., 1.);
}

void
HypothesisAnalyzer::endJob() 
{
}
