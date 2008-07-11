#include <algorithm>

#include "TMath.h"

#include "PhysicsTools/MVATrainer/interface/HelperMacros.h"
#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiJetCombMVATrainer.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiJetCombEval.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiEvtPartons.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

TtSemiJetCombMVATrainer::TtSemiJetCombMVATrainer(const edm::ParameterSet& cfg):
  leptons_   (cfg.getParameter<edm::InputTag>("leptons")),
  jets_      (cfg.getParameter<edm::InputTag>("jets")),
  matching_  (cfg.getParameter<edm::InputTag>("matching")),
  nJetsMax_  (cfg.getParameter<int>("nJetsMax")),
  lepChannel_(cfg.getParameter<int>("lepChannel"))
{
}

TtSemiJetCombMVATrainer::~TtSemiJetCombMVATrainer()
{
}

void
TtSemiJetCombMVATrainer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  mvaComputer.update<TtSemiJetCombMVARcd>("trainer", setup, "ttSemiJetCombMVA");

  // can occur in the last iteration when the 
  // MVATrainer is about to save the result
  if(!mvaComputer) return;

  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);

  edm::Handle< edm::View<reco::RecoCandidate> > leptons; 
  evt.getByLabel(leptons_, leptons);

  edm::Handle< std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);

  // skip events with no appropriate lepton candidate in
  if( leptons->empty() ) return;

  math::XYZTLorentzVector lepton = leptons->begin()->p4();

  // skip events with less jets than partons
  unsigned int nPartons = 4;
  if( jets->size() < nPartons ) return;

  edm::Handle< std::vector<int> > matching;
  // get jet-parton matching if signal channel
  if(genEvt->semiLeptonicChannel() == lepChannel_) {
    evt.getByLabel(matching_, matching);
    // skip events that were affected by the outlier 
    // rejection in the jet-parton matching
    if( std::count(matching->begin(), matching->end(), -1)>0 )
      return;
  }

  // analyze true and false jet combinations
  std::vector<int> jetIndices;
  for(unsigned int i=0; i<jets->size(); ++i) {
    if(nJetsMax_ >= nPartons && i == (unsigned int) nJetsMax_) break;
    jetIndices.push_back(i);
  }

  std::vector<int> combi;
  for(unsigned int i = 0; i < nPartons; ++i) 
    combi.push_back(i);

  do{
    // number of possible combinations from number of partons: e.g. 4! = 24
    for(unsigned int cnt = 0; cnt < TMath::Factorial( combi.size() ); ++cnt) {
      
      // take into account indistinguishability of the two jets from the hadr. W decay,
      // reduces combinatorics by a factor of 2
      if(combi[TtSemiEvtPartons::LightQ] < combi[TtSemiEvtPartons::LightQBar]) {  
	
	TtSemiJetComb jetComb(*jets, combi, lepton);
	
	bool trueCombi = true;
	if(genEvt->semiLeptonicChannel() == lepChannel_) {
	  for(unsigned int i = 0; i < matching->size(); ++i){
	    if(combi[i] != (*(matching))[i]) {
	      // not a true combination if different from matching
	      trueCombi = false;
	      break;
	    }
	  }
	}
	// no true combinations if not signal channel
	else trueCombi = false;
	
	evaluateTtSemiJetComb(mvaComputer, jetComb, true, trueCombi);

      }

      next_permutation( combi.begin() , combi.end() );
    }
  }
  while(stdcomb::next_combination( jetIndices.begin(), jetIndices.end(), combi.begin(), combi.end() ));

}

// implement the plugins for the trainer
// -> defines TtSemiJetCombMVAContainerSaveCondDB
// -> defines TtSemiJetCombMVASaveFile
// -> defines TtSemiJetCombMVATrainerLooper
MVA_TRAINER_IMPLEMENT(TtSemiJetCombMVA);
