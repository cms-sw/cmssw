#include "DataFormats/Math/interface/deltaR.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"
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
      case TtSemiLepEvtPartons::LightQ:
	jetCorrectionLevel("lightQuark").empty() ? setCandidate(jets, match[idx], lightQ_) : setCandidate(jets, match[idx], lightQ_, jetCorrectionLevel("lightQuark")); break;
      case TtSemiLepEvtPartons::LightQBar:
	jetCorrectionLevel("lightQuark").empty() ? setCandidate(jets, match[idx], lightQBar_) : setCandidate(jets, match[idx], lightQBar_, jetCorrectionLevel("lightQuark")); break;
      case TtSemiLepEvtPartons::HadB:
	jetCorrectionLevel("bJet").empty() ? setCandidate(jets, match[idx], hadronicB_) : setCandidate(jets, match[idx], hadronicB_, jetCorrectionLevel("bJet")); break;
      case TtSemiLepEvtPartons::LepB: 
	jetCorrectionLevel("bJet").empty() ? setCandidate(jets, match[idx], leptonicB_) : setCandidate(jets, match[idx], leptonicB_, jetCorrectionLevel("bJet")); break;
      }
    }
  }
 
  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  int iLepton = findMatchingLepton(evt,leps);
  if( iLepton>=0 ) setCandidate(leps, iLepton, lepton_);
  match.push_back( iLepton );

  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  if( !mets->empty() )
    setCandidate(mets, 0, neutrino_);
}

/// find index of the candidate nearest to the singleLepton of the generator event in the collection; return -1 if this fails
int
TtSemiLepHypGenMatch::findMatchingLepton(edm::Event& evt, const edm::Handle<edm::View<reco::RecoCandidate> >& leps)
{
  int genIdx=-1;

  // jump out with -1 when the collection is empty
  if( leps->empty() ) return genIdx;

  // set genEvent
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);  
  
  if( genEvt->isTtBar() && genEvt->isSemiLeptonic( leptonType( &(leps->front()) ) ) && genEvt->singleLepton() ){
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
