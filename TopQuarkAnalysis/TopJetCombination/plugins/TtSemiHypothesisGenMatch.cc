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
  std::vector<reco::ShallowCloneCandidate> leafs;
  for(std::vector<int>::const_iterator idx=match->begin(); idx!=match->end(); ++idx){    
    if( !isValid(*idx, jets) ){
      reco::ShallowCloneCandidate buffer;
      leafs.push_back( buffer );
    }
    else{
      edm::Ref<std::vector<pat::Jet> > ref=edm::Ref<std::vector<pat::Jet> >(jets, *idx);
      reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
      leafs.push_back( buffer );
    }
  }
  lightQ_   = new reco::ShallowCloneCandidate( leafs[ 0 ] );
  lightQBar_= new reco::ShallowCloneCandidate( leafs[ 1 ] );
  hadronicB_= new reco::ShallowCloneCandidate( leafs[ 2 ] );
  leptonicB_= new reco::ShallowCloneCandidate( leafs[ 3 ] );

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
