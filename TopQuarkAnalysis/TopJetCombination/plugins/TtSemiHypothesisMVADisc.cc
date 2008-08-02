#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiHypothesisMVADisc.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiEvtPartons.h"

TtSemiHypothesisMVADisc::TtSemiHypothesisMVADisc(const edm::ParameterSet& cfg):
  TtSemiHypothesis( cfg ) { }

TtSemiHypothesisMVADisc::~TtSemiHypothesisMVADisc() { }

void
TtSemiHypothesisMVADisc::buildHypo(edm::Event& evt,
				   const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
				   const edm::Handle<std::vector<pat::MET> >& mets, 
				   const edm::Handle<std::vector<pat::Jet> >& jets, 
				   std::vector<int>& match)
{
  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  for(unsigned idx=0; idx<match.size(); ++idx){    
    int ij = match[idx];   
    if( isValid( ij, jets) ){
      switch(idx){
      case TtSemiEvtPartons::LightQ:
	setCandidate(jets, ij, lightQ_); break;
      case TtSemiEvtPartons::LightQBar:
	setCandidate(jets, ij, lightQBar_); break;
      case TtSemiEvtPartons::HadB:
	setCandidate(jets, ij, hadronicB_); break;
      case TtSemiEvtPartons::LepB: 
	setCandidate(jets, ij, leptonicB_); break;
      }
    }
  }
  
  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  if( !leps->empty() )
    setCandidate(leps, 0, lepton_);
  
  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  if( !mets->empty() )
    setCandidate(mets, 0, neutrino_);
}
