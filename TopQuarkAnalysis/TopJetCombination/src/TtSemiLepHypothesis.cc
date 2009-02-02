#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"


TtSemiLepHypothesis::TtSemiLepHypothesis(const edm::ParameterSet& cfg):
  jets_(cfg.getParameter<edm::InputTag>("jets")),
  leps_(cfg.getParameter<edm::InputTag>("leps")),
  mets_(cfg.getParameter<edm::InputTag>("mets")),
  lightQ_(0), lightQBar_(0), hadronicB_(0), 
  leptonicB_(0), neutrino_(0), lepton_(0)
{
  getMatch_ = false;
  if( cfg.exists("match") ) {
    getMatch_ = true;
    match_ = cfg.getParameter<edm::InputTag>("match");
  }

  produces<std::vector<std::pair<reco::CompositeCandidate, std::vector<int> > > >();
  produces<int>("Key");
}

TtSemiLepHypothesis::~TtSemiLepHypothesis()
{
  if( lightQ_    ) delete lightQ_;
  if( lightQBar_ ) delete lightQBar_;
  if( hadronicB_ ) delete hadronicB_;
  if( leptonicB_ ) delete leptonicB_;
  if( neutrino_  ) delete neutrino_;
  if( lepton_    ) delete lepton_;
}

void
TtSemiLepHypothesis::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);
  
  edm::Handle<edm::View<reco::RecoCandidate> > leps;
  evt.getByLabel(leps_, leps);

  edm::Handle<std::vector<pat::MET> > mets;
  evt.getByLabel(mets_, mets);

  std::vector<std::vector<int> > matchVec;
  if( getMatch_ ) {
    edm::Handle<std::vector<std::vector<int> > > matchHandle;
    evt.getByLabel(match_, matchHandle);
    matchVec = *matchHandle;
  }
  else {
    std::vector<int> dummyMatch;
    for(unsigned int i = 0; i < 4; ++i) 
      dummyMatch.push_back( -1 );
    matchVec.push_back( dummyMatch );
  }

  // declare auto_ptr for products
  std::auto_ptr<std::vector<std::pair<reco::CompositeCandidate, std::vector<int> > > >
    pOut( new std::vector<std::pair<reco::CompositeCandidate, std::vector<int> > > );
  std::auto_ptr<int> pKey(new int);

  // go through given vector of jet combinations
  unsigned int idMatch = 0;
  typedef std::vector<std::vector<int> >::iterator MatchVecIterator;
  for(MatchVecIterator match = matchVec.begin(); match != matchVec.end(); ++match) {

    // reset pointers
    resetCandidates();

    // build hypothesis
    buildHypo(evt, leps, mets, jets, *match, idMatch++);

    pOut->push_back( std::make_pair(hypo(), *match) );
  }

  // feed out hyps and matches
  evt.put(pOut);

  // build and feed out key
  buildKey();
  *pKey=key();
  evt.put(pKey, "Key");
}

void
TtSemiLepHypothesis::resetCandidates()
{
  lightQ_    = 0;
  lightQBar_ = 0;
  hadronicB_ = 0;
  leptonicB_ = 0;
  neutrino_  = 0;
  lepton_    = 0;
}

reco::CompositeCandidate
TtSemiLepHypothesis::hypo()
{
  // check for sanity of the hypothesis
  if( !lightQ_ || !lightQBar_ || !hadronicB_ || 
      !leptonicB_ || !neutrino_ || !lepton_ )
    return reco::CompositeCandidate();
  
  // setup transient references
  reco::CompositeCandidate hyp, hadTop, hadW, lepTop, lepW;

  AddFourMomenta addFourMomenta;  
  // build up the top branch that decays leptonically
  lepW  .addDaughter(*lepton_,   TtSemiLepDaughter::Lep    );
  lepW  .addDaughter(*neutrino_, TtSemiLepDaughter::Nu     );
  addFourMomenta.set( lepW );
  lepTop.addDaughter( lepW,      TtSemiLepDaughter::LepW   );
  lepTop.addDaughter(*leptonicB_,TtSemiLepDaughter::LepB   );
  addFourMomenta.set( lepTop );
  
  // build up the top branch that decays hadronically
  hadW  .addDaughter(*lightQ_,   TtSemiLepDaughter::HadP   );
  hadW  .addDaughter(*lightQBar_,TtSemiLepDaughter::HadQ   );
  addFourMomenta.set( hadW );
  hadTop.addDaughter( hadW,      TtSemiLepDaughter::HadW   );
  hadTop.addDaughter(*hadronicB_,TtSemiLepDaughter::HadB   );
  addFourMomenta.set( hadTop );

  // build ttbar hypotheses
  hyp.addDaughter( lepTop,       TtSemiLepDaughter::LepTop );
  hyp.addDaughter( hadTop,       TtSemiLepDaughter::HadTop );
  addFourMomenta.set( hyp );

  return hyp;
}
