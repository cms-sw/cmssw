#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiHypothesisGenMatch.h"


TtSemiHypothesisGenMatch::TtSemiHypothesisGenMatch(const edm::ParameterSet& cfg):
  TtSemiHypothesis( cfg ) { }

TtSemiHypothesisGenMatch::~TtSemiHypothesisGenMatch() { }

void
TtSemiHypothesisGenMatch::buildHypo(const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
				    const edm::Handle<std::vector<pat::MET> >& mets, 
				    const edm::Handle<std::vector<pat::Jet> >& jets, 
				    const edm::Handle<std::vector<int> >& match)
{
  // -----------------------------------------------------
  // add jets; the order of match is Q, QBar, hadB, lepB
  // -----------------------------------------------------
  for(unsigned idx=0; idx<match->size(); ++idx){    
    if( isValid( (*match)[idx], jets) ){
      edm::Ref<std::vector<pat::Jet> > ref=edm::Ref<std::vector<pat::Jet> >(jets, (*match)[idx]);
      reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
      switch(idx){
      case 0: 
	lightQ_   = new reco::ShallowCloneCandidate( buffer ); break;
      case 1: 
	lightQBar_= new reco::ShallowCloneCandidate( buffer ); break;
      case 2: 
	hadronicB_= new reco::ShallowCloneCandidate( buffer ); break;
      case 3: 
	leptonicB_= new reco::ShallowCloneCandidate( buffer ); break;
      }
    }
  }
  
  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  if( !leps->empty() ){
    edm::Ref<edm::View<reco::RecoCandidate> > ref=edm::Ref<edm::View<reco::RecoCandidate> >(leps, 0);
    reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
    lepton_= new reco::ShallowCloneCandidate( buffer );
  }
  
  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  {
    if( !mets->empty() ){
      edm::Ref<std::vector<pat::MET> > ref=edm::Ref<std::vector<pat::MET> >(mets, 0);
      reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
      neutrino_= new reco::ShallowCloneCandidate( buffer );
    }
  }
}
