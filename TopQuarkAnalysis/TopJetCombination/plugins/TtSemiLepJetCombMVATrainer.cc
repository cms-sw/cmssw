#include "TMath.h"
#include <algorithm>

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepJetCombEval.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepJetCombMVATrainer.h"

#include "PhysicsTools/JetMCUtils/interface/combination.h"
#include "PhysicsTools/MVATrainer/interface/HelperMacros.h"

TtSemiLepJetCombMVATrainer::TtSemiLepJetCombMVATrainer(const edm::ParameterSet& cfg):
  leptons_   (cfg.getParameter<edm::InputTag>("leptons" )),
  jets_      (cfg.getParameter<edm::InputTag>("jets"    )),
  mets_      (cfg.getParameter<edm::InputTag>("mets"    )),
  matching_  (cfg.getParameter<edm::InputTag>("matching")),
  maxNJets_  (cfg.getParameter<int>          ("maxNJets")),
  leptonType_(readLeptonType(cfg.getParameter<std::string>("leptonType")))
{
}

TtSemiLepJetCombMVATrainer::~TtSemiLepJetCombMVATrainer()
{
}

void 
TtSemiLepJetCombMVATrainer::beginJob(const edm::EventSetup&)
{
  for(unsigned int i = 0; i < 5; i++)
    nEvents[i] = 0;
}

void
TtSemiLepJetCombMVATrainer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  mvaComputer.update<TtSemiLepJetCombMVARcd>("trainer", setup, "ttSemiLepJetCombMVA");

  // can occur in the last iteration when the 
  // MVATrainer is about to save the result
  if(!mvaComputer) return;

  nEvents[0]++;

  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);

  edm::Handle< edm::View<reco::RecoCandidate> > leptons; 
  evt.getByLabel(leptons_, leptons);

  // skip events with no appropriate lepton candidate
  if( leptons->empty() ) return;

  nEvents[1]++;

  const math::XYZTLorentzVector lepton = leptons->begin()->p4();

  edm::Handle< std::vector<pat::MET> > mets;
  evt.getByLabel(mets_, mets);

  // skip events with empty METs vector
  if( mets->empty() ) return;

  nEvents[2]++;

  const pat::MET *met = &(*mets)[0];

  edm::Handle< std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);

  // skip events with less jets than partons
  unsigned int nPartons = 4;
  if( jets->size() < nPartons ) return;

  nEvents[3]++;

  edm::Handle< std::vector< std::vector<int> > > matchingHandle;
  std::vector<int> matching;
  // get jet-parton matching if signal channel
  if(genEvt->semiLeptonicChannel() == leptonType_) {
    evt.getByLabel(matching_, matchingHandle);
    if( matchingHandle->empty() )
      throw cms::Exception("EmptyProduct")
	<< "Empty vector from jet-parton matching. This should not happen! \n";
    matching = *(matchingHandle->begin());
    if(matching.size() < nPartons) return;
    // skip events that were affected by the outlier 
    // rejection in the jet-parton matching
    for(unsigned int i = 0; i < matching.size(); ++i)
      if(matching[i] < 0 || matching[i] >= (int)jets->size())
	return;
  }
  // use dummy for matching if not signal channel
  else
    for(unsigned int i = 0; i < nPartons; i++)
      matching.push_back( -1 );

  nEvents[4]++;

  // take into account indistinguishability of the two jets from the hadr. W decay,
  // reduces combinatorics by a factor of 2
  if(matching[TtSemiLepEvtPartons::LightQ] > matching[TtSemiLepEvtPartons::LightQBar]) {
    int iTemp = matching[TtSemiLepEvtPartons::LightQ];
    matching[TtSemiLepEvtPartons::LightQ] = matching[TtSemiLepEvtPartons::LightQBar];
    matching[TtSemiLepEvtPartons::LightQBar] = iTemp;
  }

  std::vector<int> jetIndices;
  for(unsigned int i=0; i<jets->size(); ++i) {
    if(maxNJets_ >= (int) nPartons && i == (unsigned int) maxNJets_) break;
    jetIndices.push_back(i);
  }

  std::vector<int> combi;
  for(unsigned int i = 0; i < nPartons; ++i) 
    combi.push_back(i);

  // use a ValueList to collect all variables for the MVAComputer
  PhysicsTools::Variable::ValueList values;

  // set target variable to false like for background
  // this is necessary when passing all combinations at once
  values.add( PhysicsTools::Variable::Value(PhysicsTools::MVATrainer::kTargetId, false) );

  do {
    // number of possible combinations from number of partons: e.g. 4! = 24
    for(unsigned int cnt = 0; cnt < TMath::Factorial( combi.size() ); ++cnt) {

      // take into account indistinguishability of the two jets from the hadr. W decay,
      // reduces combinatorics by a factor of 2
      if(combi[TtSemiLepEvtPartons::LightQ] < combi[TtSemiLepEvtPartons::LightQBar]) {
	
	TtSemiLepJetComb jetComb(*jets, combi, lepton, *met);

	bool trueCombi = false;
	// true combination only if signal channel
	// and in agreement with matching
	if(genEvt->semiLeptonicChannel()==leptonType_ && combi==matching)
	  trueCombi = true;

	// feed MVA input variables for this jetComb into the ValueList
	values.add("target", trueCombi);
	evaluateTtSemiLepJetComb(values, jetComb);

      }

      next_permutation( combi.begin() , combi.end() );
    }
  }
  while(stdcomb::next_combination( jetIndices.begin(), jetIndices.end(), combi.begin(), combi.end() ));

  // pass MVA input variables for all jet combinations in this event
  // to the MVAComputer for training
  mvaComputer->eval( values );
}

void
TtSemiLepJetCombMVATrainer::endJob() 
{
  edm::LogInfo log("TtSemiLepJetCombMVATrainer");
  log << "Number of events... \n"
      << "...passed to the trainer                   : " << std::setw(7) << nEvents[0] << "\n"
      << "...rejected since no lepton candidate      : " << std::setw(7) << nEvents[0]-nEvents[1] << "\n"
      << "...rejected since no MET object            : " << std::setw(7) << nEvents[1]-nEvents[2] << "\n"
      << "...rejected since not enough jets          : " << std::setw(7) << nEvents[2]-nEvents[3] << "\n"
      << "...rejected due to bad jet-parton matching : " << std::setw(7) << nEvents[3]-nEvents[4] << "\n"
      << "...accepted for training                   : " << std::setw(7) << nEvents[4] << "\n";
}

WDecay::LeptonType
TtSemiLepJetCombMVATrainer::readLeptonType(const std::string& str)
{
  if     (str == "kElec") return WDecay::kElec;
  else if(str == "kMuon") return WDecay::kMuon;
  else if(str == "kTau" ) return WDecay::kTau;
  else throw cms::Exception("Configuration")
    << "Chosen leptonType is not supported: " << str << "\n";
}

// implement the plugins for the trainer
// -> defines TtSemiLepJetCombMVAContainerSaveCondDB
// -> defines TtSemiLepJetCombMVASaveFile
// -> defines TtSemiLepJetCombMVATrainerLooper
MVA_TRAINER_IMPLEMENT(TtSemiLepJetCombMVA);
