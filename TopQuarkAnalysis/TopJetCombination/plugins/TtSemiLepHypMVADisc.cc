#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypMVADisc.h"


TtSemiLepHypMVADisc::TtSemiLepHypMVADisc(const edm::ParameterSet& cfg):
  TtSemiLepHypothesis( cfg ) { }

TtSemiLepHypMVADisc::~TtSemiLepHypMVADisc() { }

void
TtSemiLepHypMVADisc::buildHypo(edm::Event& evt,
			       const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
			       const edm::Handle<std::vector<pat::MET> >& mets, 
			       const edm::Handle<std::vector<pat::Jet> >& jets, 
			       std::vector<int>& match, const unsigned int iComb)
{
  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  //
  // FIXME:
  // template does not work anymore with new compiler
  // we need to fix this as soon as test data arrive
  //
  for(unsigned idx=0; idx<match.size(); ++idx){
    if( isValid(match[idx], jets) ){
      switch(idx){
      case TtSemiLepEvtPartons::LightQ:
	//setCandidate(jets, match[idx], lightQ_); 
	break;
      case TtSemiLepEvtPartons::LightQBar:
	//setCandidate(jets, match[idx], lightQBar_); 
	break;
      case TtSemiLepEvtPartons::HadB:
	//setCandidate(jets, match[idx], hadronicB_); 
	break;
      case TtSemiLepEvtPartons::LepB: 
	//setCandidate(jets, match[idx], leptonicB_); 
	break;
      }
    }
  }

  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  //
  // FIXME:
  // template does not work anymore with new compiler
  // we need to fix this as soon as test data arrive
  //
  if( !leps->empty() ){
    //setCandidate(leps, 0, lepton_);
  }
  match.push_back( 0 );
  
  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  //
  // FIXME:
  // template does not work anymore with new compiler
  // we need to fix this as soon as test data arrive
  //
  if( !mets->empty() ){
    //setCandidate(mets, 0, neutrino_);
  }
}
