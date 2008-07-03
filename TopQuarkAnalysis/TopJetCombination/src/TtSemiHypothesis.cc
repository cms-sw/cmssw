#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiHypothesis.h"


TtSemiHypothesis::TtSemiHypothesis(const edm::ParameterSet& cfg):
  jets_ (cfg.getParameter<edm::InputTag>("jets" )),
  leps_ (cfg.getParameter<edm::InputTag>("leps" )),
  mets_ (cfg.getParameter<edm::InputTag>("mets" )),
  match_(cfg.getParameter<edm::InputTag>("match")),
  lightQ_(0), lightQBar_(0), hadronicB_(0), 
  leptonicB_(0), neutrino_(0), lepton_(0)
{
  produces<reco::CompositeCandidate>();
  produces<int>("Key");
}

TtSemiHypothesis::~TtSemiHypothesis()
{
  if( lightQ_   ) delete lightQ_;
  if( lightQBar_) delete lightQBar_;
  if( hadronicB_) delete hadronicB_;
  if( leptonicB_) delete leptonicB_;
  if( neutrino_ ) delete neutrino_;
  if( lepton_   ) delete lepton_;
}

void
TtSemiHypothesis::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);
  
  edm::Handle<edm::View<reco::RecoCandidate> > leps;
  evt.getByLabel(leps_, leps);

  edm::Handle<std::vector<pat::MET> > mets;
  evt.getByLabel(mets_, mets);

  edm::Handle<std::vector<int> > match;
  evt.getByLabel(match_, match);

  // feed out hyp
  std::auto_ptr<reco::CompositeCandidate> pOut(new reco::CompositeCandidate);
  buildHypo(leps, mets, jets, match);
  *pOut=hypo();
  evt.put(pOut);
  
  // feed out key
  std::auto_ptr<int> pKey(new int);
  buildKey();
  *pKey=key();
  evt.put(pKey, "Key");
}

reco::CompositeCandidate
TtSemiHypothesis::hypo()
{
  // check for sanity of the hypothesis
  if( !lightQ_ || !lightQBar_ || !hadronicB_ || 
      !leptonicB_ || !neutrino_ || !lepton_ )
    return reco::CompositeCandidate();
  
  // setup transient references
  reco::CompositeCandidate hyp, hadTop, hadW, lepTop, lepW;

  AddFourMomenta addFourMomenta;  
  // build up the top branch that decays leptonically
  lepW  .addDaughter(*neutrino_, TtSemiDaughter::Nu     );
  lepW  .addDaughter(*lepton_,   TtSemiDaughter::Lep    );
  addFourMomenta.set( lepW );
  lepTop.addDaughter( lepW,      TtSemiDaughter::LepW   );
  lepTop.addDaughter(*leptonicB_,TtSemiDaughter::LepB   );
  addFourMomenta.set( lepTop );
  
  // build up the top branch that decays hadronically
  hadW  .addDaughter(*lightQ_,   TtSemiDaughter::HadQ   );
  hadW  .addDaughter(*lightQBar_,TtSemiDaughter::HadP   );
  addFourMomenta.set( hadW );
  hadTop.addDaughter( hadW,      TtSemiDaughter::HadW   );
  hadTop.addDaughter(*hadronicB_,TtSemiDaughter::HadB   );
  addFourMomenta.set( hadTop );

  // build ttbar hypotheses
  hyp.addDaughter( lepTop,       TtSemiDaughter::LepTop );
  hyp.addDaughter( hadTop,       TtSemiDaughter::HadTop );
  addFourMomenta.set( hyp );

  return hyp;
}
