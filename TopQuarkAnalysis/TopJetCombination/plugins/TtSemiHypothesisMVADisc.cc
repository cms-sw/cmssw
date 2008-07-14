#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiHypothesisMVADisc.h"


TtSemiHypothesisMVADisc::TtSemiHypothesisMVADisc(const edm::ParameterSet& cfg):
  TtSemiHypothesis( cfg ) { }

TtSemiHypothesisMVADisc::~TtSemiHypothesisMVADisc() { }

void
TtSemiHypothesisMVADisc::buildHypo(const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
				   const edm::Handle<std::vector<pat::MET> >& mets, 
				   const edm::Handle<std::vector<pat::Jet> >& jets, 
				   std::vector<int>& match)
{
  // -----------------------------------------------------
  // add jets; the order of match is Q, QBar, hadB, lepB
  // -----------------------------------------------------
  for(unsigned idx=0; idx<match.size(); ++idx){    
    int ij = match[idx];   
    if( isValid( ij, jets) ){
      switch(idx){
      case 0: 
	{
	  edm::Ptr<pat::Jet> jet = edm::Ptr<pat::Jet>(jets, ij);
	  lightQ_   = new reco::ShallowClonePtrCandidate( jet, jet->charge(), jet->p4(), jet->vertex() ); 
	  break;
	}
      case 1:
	{ 
	  edm::Ptr<pat::Jet> jet = edm::Ptr<pat::Jet>(jets, ij);
	  lightQBar_= new reco::ShallowClonePtrCandidate( jet, jet->charge(), jet->p4(), jet->vertex() ); 
	  break;
	}
      case 2:
	{
	  edm::Ptr<pat::Jet> jet = edm::Ptr<pat::Jet>(jets, ij);
	  hadronicB_= new reco::ShallowClonePtrCandidate( jet, jet->charge(), jet->p4(), jet->vertex() ); 
	  break;
	}
      case 3:
	{
	  edm::Ptr<pat::Jet> jet = edm::Ptr<pat::Jet>(jets, ij);
	  leptonicB_= new reco::ShallowClonePtrCandidate( jet, jet->charge(), jet->p4(), jet->vertex() ); 
	  break;
	}
      }
    }
  }

  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  if( !leps->empty() ){
    edm::Ptr<reco::RecoCandidate> lep = edm::Ptr<reco::RecoCandidate>(leps, 0);
    lepton_= new reco::ShallowClonePtrCandidate( lep, lep->charge(), lep->p4(), lep->vertex() );
  }
  
  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  if( !mets->empty() ){
    edm::Ptr<pat::MET> met = edm::Ptr<pat::MET>(mets, 0);
    neutrino_= new reco::ShallowClonePtrCandidate( met, met->charge(), met->p4(), met->vertex() );
  }
}
