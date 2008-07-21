#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiHypothesisMaxSumPtWMass.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiEvtPartons.h"

TtSemiHypothesisMaxSumPtWMass::TtSemiHypothesisMaxSumPtWMass(const edm::ParameterSet& cfg):
  TtSemiHypothesis( cfg ),  
  maxNJets_(cfg.getParameter<unsigned>("nJetsMax" ))
{ 

}

TtSemiHypothesisMaxSumPtWMass::~TtSemiHypothesisMaxSumPtWMass() { }

void
TtSemiHypothesisMaxSumPtWMass::buildHypo(edm::Event& evt,
					 const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
					 const edm::Handle<std::vector<pat::MET> >& mets, 
					 const edm::Handle<std::vector<pat::Jet> >& jets, 
					 std::vector<int>& match)
{
  if(leps->empty() || mets->empty() || jets->size()<maxNJets_ || maxNJets_<4){
    // create empty hypothesis
    return;
  }

  for(unsigned int i=0; i<4; ++i)
    match.push_back(-1);

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
    int ij = closestToWMassIndices[0];  
    edm::Ptr<pat::Jet> jet = edm::Ptr<pat::Jet>(jets, ij);
    lightQ_= new reco::ShallowClonePtrCandidate( jet, jet->charge(), jet->p4(), jet->vertex() );
    match[TtSemiEvtPartons::LightQ] = closestToWMassIndices[0];
  }

  if( isValid(closestToWMassIndices[1], jets) ){
    int ij = closestToWMassIndices[1];  
    edm::Ptr<pat::Jet> jet = edm::Ptr<pat::Jet>(jets, ij);
    lightQBar_= new reco::ShallowClonePtrCandidate( jet, jet->charge(), jet->p4(), jet->vertex() );
    match[TtSemiEvtPartons::LightQBar] = closestToWMassIndices[1];
  }

  for(unsigned idx=0; idx<maxPtIndices.size(); ++idx){
    // if this idx is not yet contained in the list of W mass candidates...
    if( std::find( closestToWMassIndices.begin(), closestToWMassIndices.end(), maxPtIndices[idx]) == closestToWMassIndices.end() ){
      // ...and if it is valid
      if( isValid(maxPtIndices[idx], jets) ){
	int ij = maxPtIndices[idx];  
	edm::Ptr<pat::Jet> jet = edm::Ptr<pat::Jet>(jets, ij);
	hadronicB_= new reco::ShallowClonePtrCandidate( jet, jet->charge(), jet->p4(), jet->vertex() );
	match[TtSemiEvtPartons::HadB] = maxPtIndices[idx];
	break; // there should be no other cadidates!
      }
    }
  }

  if( isValid(lepB, jets) ){
    int ij = lepB;  
    edm::Ptr<pat::Jet> jet = edm::Ptr<pat::Jet>(jets, ij);
    leptonicB_= new reco::ShallowClonePtrCandidate( jet, jet->charge(), jet->p4(), jet->vertex() );
    match[TtSemiEvtPartons::LepB] = lepB;
  }

  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  edm::Ptr<reco::RecoCandidate> lep = edm::Ptr<reco::RecoCandidate>(leps, 0);
  lepton_= new reco::ShallowClonePtrCandidate( lep, lep->charge(), lep->p4(), lep->vertex() );
  
  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  edm::Ptr<pat::MET> met = edm::Ptr<pat::MET>(mets, 0);
  neutrino_= new reco::ShallowClonePtrCandidate( met, met->charge(), met->p4(), met->vertex() );
}
