#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiHypothesisMaxSumPtWMass.h"


TtSemiHypothesisMaxSumPtWMass::TtSemiHypothesisMaxSumPtWMass(const edm::ParameterSet& cfg):
  TtSemiHypothesis( cfg ),  
  maxNJets_(cfg.getParameter<unsigned>("maxNJets" ))
{ 

}

TtSemiHypothesisMaxSumPtWMass::~TtSemiHypothesisMaxSumPtWMass() { }

void
TtSemiHypothesisMaxSumPtWMass::buildHypo(const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
				    const edm::Handle<std::vector<pat::MET> >& mets, 
				    const edm::Handle<std::vector<pat::Jet> >& jets, 
				    const edm::Handle<std::vector<int> >& match)
{
  if(jets->size()<maxNJets_ || maxNJets_<4){
    // create empty hypothesis
    return;
  }
  
  // -----------------------------------------------------
  // associate those jets with maximum pt of the vectorial 
  // sum to the hadronic decay chain
  // -----------------------------------------------------
  double maxPt=-1.;
  std::vector<unsigned> maxPtIndices;
  for(unsigned idx=0; idx<maxNJets_; ++idx){
    for(unsigned jdx=0; jdx<maxNJets_; ++jdx){
      if( jdx==idx ) continue;
      for(unsigned kdx=0; kdx<maxNJets_; ++kdx){
	if( kdx==jdx || kdx==idx ) continue;
	reco::Particle::LorentzVector sum = 
	  (*jets)[idx].p4()+
	  (*jets)[jdx].p4()+
	  (*jets)[kdx].p4();
	if( maxPt<0. || maxPt<sum.pt() ){
	  maxPt=sum.pt();
	  maxPtIndices.clear();
	  maxPtIndices.push_back(idx);
	  maxPtIndices.push_back(jdx);
	  maxPtIndices.push_back(kdx);
	}
      }
    }
  }
  
  // -----------------------------------------------------
  // associate those jets that get closest to the W mass
  // with their invariant mass to the W boson
  // -----------------------------------------------------
  double wDist =-1.;
  double wMass = 80.413;
  std::vector<unsigned> closestToWMassIndices;
  for(unsigned idx=0; idx<maxPtIndices.size(); ++idx){  
    for(unsigned jdx=0; jdx<maxPtIndices.size(); ++jdx){  
      if( jdx==idx ) continue;
      reco::Particle::LorentzVector sum = 
	(*jets)[idx].p4()+
	(*jets)[jdx].p4();
      if( wDist<0. || wDist<(sum.mass()-wMass) ){
	wDist=sum.mass()-wMass;
	closestToWMassIndices.clear();
	closestToWMassIndices.push_back(maxPtIndices[idx]);
	closestToWMassIndices.push_back(maxPtIndices[jdx]);
      }
    }
  }
  
  // -----------------------------------------------------
  // associate the remaining jet with maximum pt of the   
  // vectorial sum with the leading lepton with the 
  // leptonic decay chain
  // -----------------------------------------------------
  maxPt=-1.;
  unsigned lepB=0;
  for(unsigned idx=0; idx<maxNJets_; ++idx){
    for(unsigned jdx=0; jdx<maxPtIndices.size(); ++jdx){
      // make sure it's not used up already from  
      // the hadronic decay chain
      if(idx != maxPtIndices[jdx] ){
	reco::Particle::LorentzVector sum = 
	  (*jets)[idx].p4()+(*leps)[ 0 ].p4();
	if( maxPt<0. || maxPt<sum.pt() ){
	  maxPt=sum.pt();
	  lepB=idx;
	}
      }
    }
  }

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  if( isValid(closestToWMassIndices[0], jets) ){
    edm::Ref<std::vector<pat::Jet> > ref=edm::Ref<std::vector<pat::Jet> >(jets, closestToWMassIndices[0]);
    reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
    lightQ_= new reco::ShallowCloneCandidate( buffer );
  }

  if( isValid(closestToWMassIndices[1], jets) ){
    edm::Ref<std::vector<pat::Jet> > ref=edm::Ref<std::vector<pat::Jet> >(jets, closestToWMassIndices[1]);
    reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
    lightQBar_= new reco::ShallowCloneCandidate( buffer );
  }

  for(unsigned idx=0; idx<maxPtIndices.size(); ++idx){
    // if this idx is not yet contained in the list of W mass candidates...
    if( std::find( closestToWMassIndices.begin(), closestToWMassIndices.end(), maxPtIndices[idx]) == closestToWMassIndices.end() ){
      // ...and if it is valid
      if( isValid(maxPtIndices[idx], jets) ){
	edm::Ref<std::vector<pat::Jet> > ref=edm::Ref<std::vector<pat::Jet> >(jets, maxPtIndices[idx]);
	reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
	hadronicB_= new reco::ShallowCloneCandidate( buffer );
	break; // there should be no other cadidates!
      }
    }
  }

  if( isValid(lepB, jets) ){
    edm::Ref<std::vector<pat::Jet> > ref=edm::Ref<std::vector<pat::Jet> >(jets, lepB);
    reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
    leptonicB_= new reco::ShallowCloneCandidate( buffer );
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
  if( !mets->empty() ){
    edm::Ref<std::vector<pat::MET> > ref=edm::Ref<std::vector<pat::MET> >(mets, 0);
    reco::ShallowCloneCandidate buffer(reco::CandidateBaseRef( ref ), ref->charge(), ref->p4(), ref->vertex());
    neutrino_= new reco::ShallowCloneCandidate( buffer );
  }
}
