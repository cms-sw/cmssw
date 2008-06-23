#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvent.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiHypothesisGenMatch.h"


TtSemiHypothesisGenMatch::TtSemiHypothesisGenMatch(const edm::ParameterSet& cfg):
  jets_ (cfg.getParameter<edm::InputTag>("jets" )),
  leps_ (cfg.getParameter<edm::InputTag>("leps" )),
  mets_ (cfg.getParameter<edm::InputTag>("mets" )),
  match_(cfg.getParameter<edm::InputTag>("match"))
{
  // produces an event hypothesis based on 
  // generator matching plus corresponding key
  produces<reco::NamedCompositeCandidate>();
  produces<int>("Key");
}

TtSemiHypothesisGenMatch::~TtSemiHypothesisGenMatch()
{
}

void
TtSemiHypothesisGenMatch::produce(edm::Event& evt, const edm::EventSetup& setup)
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
  std::auto_ptr<reco::NamedCompositeCandidate> pOut(new reco::NamedCompositeCandidate);
  *pOut=buildHypo(jets, leps, mets, match);
  evt.put(pOut);
  
  // feed out key
  std::auto_ptr<int> pKey(new int);
  *pKey=TtSemiEvent::kGenMatch;
  evt.put(pKey, "Key");
}

reco::NamedCompositeCandidate
TtSemiHypothesisGenMatch::buildHypo(const edm::Handle<std::vector<pat::Jet>  >& jets, 
				    const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
				    const edm::Handle<std::vector<pat::MET>  >& mets, 
				    const edm::Handle<std::vector<int> >& match)
{
  // --------------------------------------------
  // get references to leaf nodes in order:
  // >> Q, Qbar, HadB, LepB, Lep, Neutrino <<
  // --------------------------------------------
  std::vector<reco::ShallowCloneCandidate> leafs;
  for(std::vector<int>::const_iterator idx=match->begin(); idx!=match->end(); ++idx){    
    if( !isValid(*idx, jets) )
      return reco::NamedCompositeCandidate();
    
    edm::Ref<std::vector<pat::Jet> > ref=edm::Ref<std::vector<pat::Jet> >(jets, *idx);
    reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
    leafs.push_back( buffer );
  }
  // add lepton
  {
    if( leps->empty() ) 
      return reco::NamedCompositeCandidate();

    edm::Ref<edm::View<reco::RecoCandidate> > ref=edm::Ref<edm::View<reco::RecoCandidate> >(leps, 0);
    reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
    leafs.push_back( buffer );
  }
  // add neutrino
  {
    if( mets->empty() ) 
      return reco::NamedCompositeCandidate();
    
    edm::Ref<std::vector<pat::MET> > ref=edm::Ref<std::vector<pat::MET> >(mets, 0);
    reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
    leafs.push_back( buffer );
  }
  return fillHypo( leafs );
}

reco::NamedCompositeCandidate
TtSemiHypothesisGenMatch::fillHypo(std::vector<reco::ShallowCloneCandidate>& leafs)
{
  // setup transient references
  reco::NamedCompositeCandidate hyp, hadTop, hadW, lepTop, lepW;

  AddFourMomenta addFourMomenta;  
  // build up leptonically decaying top branch from back to front
  lepW  .addDaughter( leafs[leafs.size()-1], TtSemiDaughter::Nu   );
  lepW  .addDaughter( leafs[leafs.size()-2], TtSemiDaughter::Lep  );
  addFourMomenta.set( lepW );
  lepTop.addDaughter( lepW,                  TtSemiDaughter::LepW );
  lepTop.addDaughter( leafs[leafs.size()-3], TtSemiDaughter::LepB );
  addFourMomenta.set( lepTop );
  
  // build up hadronically decaying top branch from front to back
  hadW  .addDaughter( leafs[0],              TtSemiDaughter::HadQ );
  hadW  .addDaughter( leafs[1],              TtSemiDaughter::HadP );
  addFourMomenta.set( hadW );
  hadTop.addDaughter( hadW,                  TtSemiDaughter::HadW );
  hadTop.addDaughter( leafs[2],              TtSemiDaughter::HadB );
  addFourMomenta.set( hadTop );
  
  // build ttbar hypotheses
  hyp.addDaughter( lepTop, TtSemiDaughter::LepTop );
  hyp.addDaughter( hadTop, TtSemiDaughter::HadTop );
  addFourMomenta.set( hyp );
  
  return hyp;
}
