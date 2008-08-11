#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepMVADisc.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiEvtPartons.h"

TtSemiLepMVADisc::TtSemiLepMVADisc(const edm::ParameterSet& cfg):
  TtSemiLepHypothesis( cfg ) { }

TtSemiLepMVADisc::~TtSemiLepMVADisc() { }

void
TtSemiLepMVADisc::buildHypo(edm::Event& evt,
				   const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
				   const edm::Handle<std::vector<pat::MET> >& mets, 
				   const edm::Handle<std::vector<pat::Jet> >& jets, 
				   std::vector<int>& match)
{
  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  for(unsigned idx=0; idx<match.size(); ++idx){
    if( isValid(match[idx], jets) ){
      switch(idx){
      case TtSemiEvtPartons::LightQ:
	setCandidate(jets, match[idx], lightQ_); break;
      case TtSemiEvtPartons::LightQBar:
	setCandidate(jets, match[idx], lightQBar_); break;
      case TtSemiEvtPartons::HadB:
	setCandidate(jets, match[idx], hadronicB_); break;
      case TtSemiEvtPartons::LepB: 
	setCandidate(jets, match[idx], leptonicB_); break;
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
