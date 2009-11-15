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
  // get genEvent (to distinguish between uds and c quarks
  // and for the lepton matching)
  // -----------------------------------------------------
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);  

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  for(unsigned idx=0; idx<match.size(); ++idx){
    if( isValid(match[idx], jets) ){
      switch(idx){
      case TtSemiLepEvtPartons::LightQ:
	if( abs(genEvt->hadronicDecayQuark()->pdgId())==4 )
	  setCandidate(jets, match[idx], lightQ_, jetCorrectionLevel("cQuark"));
	else
	  setCandidate(jets, match[idx], lightQ_, jetCorrectionLevel("udsQuark"));
	break;
      case TtSemiLepEvtPartons::LightQBar:
	if( abs(genEvt->hadronicDecayQuarkBar()->pdgId())==4 )
	  setCandidate(jets, match[idx], lightQBar_, jetCorrectionLevel("cQuark"));
	else
	  setCandidate(jets, match[idx], lightQBar_, jetCorrectionLevel("udsQuark"));
	break;
      case TtSemiLepEvtPartons::HadB:
	setCandidate(jets, match[idx], hadronicB_, jetCorrectionLevel("bQuark")); break;
      case TtSemiLepEvtPartons::LepB: 
	setCandidate(jets, match[idx], leptonicB_, jetCorrectionLevel("bQuark")); break;
      }
    }
  }
 
  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  int iLepton = findMatchingLepton(genEvt, leps);
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
TtSemiLepHypGenMatch::findMatchingLepton(const edm::Handle<TtGenEvent>& genEvt,
					 const edm::Handle<edm::View<reco::RecoCandidate> >& leps)
{
  int genIdx=-1;

  // jump out with -1 when the collection is empty
  if( leps->empty() ) return genIdx;
  
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
