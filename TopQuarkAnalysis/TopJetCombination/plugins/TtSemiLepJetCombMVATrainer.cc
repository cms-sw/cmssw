#include "TMath.h"
#include <algorithm>

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepJetCombEval.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepJetCombMVATrainer.h"

#include "PhysicsTools/JetMCUtils/interface/combination.h"
#include "PhysicsTools/MVATrainer/interface/HelperMacros.h"


TtSemiLepJetCombMVATrainer::TtSemiLepJetCombMVATrainer(const edm::ParameterSet& cfg):
  leptons_   (cfg.getParameter<edm::InputTag>("leptons")),
  jets_      (cfg.getParameter<edm::InputTag>("jets")),
  mets_      (cfg.getParameter<edm::InputTag>("mets")),
  matching_  (cfg.getParameter<edm::InputTag>("matching")),
  maxNJets_  (cfg.getParameter<int>("maxNJets")),
  lepChannel_(cfg.getParameter<int>("lepChannel"))
{
}

TtSemiLepJetCombMVATrainer::~TtSemiLepJetCombMVATrainer()
{
}

void
TtSemiLepJetCombMVATrainer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  mvaComputer.update<TtSemiLepJetCombMVARcd>("trainer", setup, "ttSemiLepJetCombMVA");

  // can occur in the last iteration when the 
  // MVATrainer is about to save the result
  if(!mvaComputer) return;

  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);

  edm::Handle< edm::View<reco::RecoCandidate> > leptons; 
  evt.getByLabel(leptons_, leptons);

  edm::Handle< std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);

  edm::Handle< std::vector<pat::MET> > mets;
  evt.getByLabel(mets_, mets);

  //skip events with no appropriate lepton candidate
  if( leptons->empty() ) return;
  
  const math::XYZTLorentzVector lepton = leptons->begin()->p4();
  
  //skip events with empty METs vector
  if( mets->empty() ) return;

  const pat::MET *met = &(*mets)[0];

  // skip events with less jets than partons
  unsigned int nPartons = 4;
  if( jets->size() < nPartons ) return;

  edm::Handle< std::vector< std::vector<int> > > matchingHandle;
  std::vector<int> matching;
  // get jet-parton matching if signal channel
  if(genEvt->semiLeptonicChannel() == lepChannel_) {
    evt.getByLabel(matching_, matchingHandle);
    matching = *(matchingHandle->begin());
    // skip events that were affected by the outlier 
    // rejection in the jet-parton matching
    for(unsigned int i = 0; i < matching.size(); ++i)
      if(matching[i] < 0 || matching[i] >= (int)jets->size())
	return;
  }

  // analyze true and false jet combinations
  std::vector<int> jetIndices;
  for(unsigned int i=0; i<jets->size(); ++i) {
    if(maxNJets_ >= nPartons && i == (unsigned int) maxNJets_) break;
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
      if(combi[TtSemiLepEvtPartons::LightQ] < combi[TtSemiLepEvtPartons::LightQBar]) {  
	
	TtSemiLepJetComb jetComb(*jets, combi, lepton, *met);

	
	bool trueCombi = true;
	if(genEvt->semiLeptonicChannel() == lepChannel_) {
	  for(unsigned int i = 0; i < matching.size(); ++i){
	    if(combi[i] != matching[i]) {
	      // not a true combination if different from matching
	      trueCombi = false;
	      break;
	    }
	  }
	}
	// no true combinations if not signal channel
	else trueCombi = false;
	
	evaluateTtSemiLepJetComb(mvaComputer, jetComb, true, trueCombi);

      }

      next_permutation( combi.begin() , combi.end() );
    }
  }
  while(stdcomb::next_combination( jetIndices.begin(), jetIndices.end(), combi.begin(), combi.end() ));

}

// implement the plugins for the trainer
// -> defines TtSemiLepJetCombMVAContainerSaveCondDB
// -> defines TtSemiLepJetCombMVASaveFile
// -> defines TtSemiLepJetCombMVATrainerLooper
MVA_TRAINER_IMPLEMENT(TtSemiLepJetCombMVA);
