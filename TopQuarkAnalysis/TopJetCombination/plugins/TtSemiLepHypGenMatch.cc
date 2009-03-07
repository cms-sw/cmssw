#include "DataFormats/Math/interface/deltaR.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypGenMatch.h"


TtSemiLepHypGenMatch::TtSemiLepHypGenMatch(const edm::ParameterSet& cfg):
  TtSemiLepHypothesis( cfg ) { }

TtSemiLepHypGenMatch::~TtSemiLepHypGenMatch() { }

void
TtSemiLepHypGenMatch::buildHypo(edm::Event& evt,
				const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
				const edm::Handle<std::vector<pat::MET> >& mets, 
				const edm::Handle<std::vector<pat::Jet> >& jets, 
				std::vector<int>& match, const unsigned int iComb)
{
  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  for(unsigned idx=0; idx<match.size(); ++idx){
    if( isValid(match[idx], jets) ){
      switch(idx){
	//
	// FIXME:
	// template does not work anymore with new compiler
	// we need to fix this as soon as test data arrive
	//
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
  if( !leps->empty() ){
    int iLepton = findMatchingLepton(evt,leps);
    //
    // FIXME:
    // template does not work anymore with new compiler
    // we need to fix this as soon as test data arrive
    //
    if( iLepton>=0 ){
      //setCandidate(leps, iLepton, lepton_);
    }
    match.push_back( iLepton );
  }
  else match.push_back( -1 );

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

int
TtSemiLepHypGenMatch::findMatchingLepton(edm::Event& evt, const edm::Handle<edm::View<reco::RecoCandidate> >& leps)
{
  int genIdx=-1;

  // set genEvent
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);  
  
  if( genEvt->isTtBar() && genEvt->isSemiLeptonic() && genEvt->singleLepton() ){
    double minDR=-1;
    for(unsigned i=0; i<leps->size(); ++i){
      double dR = deltaR(genEvt->singleLepton()->eta(), genEvt->singleLepton()->phi(), (*leps)[i].eta(), (*leps)[i].phi());
      if(minDR<0 || dR<minDR){
	minDR=dR;
	genIdx=i;
      }
    }
  }
  return genIdx;
}
