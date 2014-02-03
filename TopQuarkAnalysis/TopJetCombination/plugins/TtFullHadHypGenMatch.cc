#include "TopQuarkAnalysis/TopJetCombination/plugins/TtFullHadHypGenMatch.h"
#include "AnalysisDataFormats/TopObjects/interface/TtFullHadEvtPartons.h"

TtFullHadHypGenMatch::TtFullHadHypGenMatch(const edm::ParameterSet& cfg):
  TtFullHadHypothesis( cfg ) 
{  
}

TtFullHadHypGenMatch::~TtFullHadHypGenMatch() { }

void
TtFullHadHypGenMatch::buildHypo(edm::Event& evt,
			        const edm::Handle<std::vector<pat::Jet> >& jets, 
			        std::vector<int>& match,
				const unsigned int iComb)
{
  // -----------------------------------------------------
  // get genEvent (to distinguish between uds and c quarks)
  // -----------------------------------------------------
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  for(unsigned idx=0; idx<match.size(); ++idx){
    if( isValid(match[idx], jets) ){
      switch(idx){
      case TtFullHadEvtPartons::LightQ:
	if( std::abs(genEvt->daughterQuarkOfWPlus()->pdgId())==4 )
	  setCandidate(jets, match[idx], lightQ_   , jetCorrectionLevel("cQuark"));
	else
	  setCandidate(jets, match[idx], lightQ_   , jetCorrectionLevel("udsQuark"));
	break;
      case TtFullHadEvtPartons::LightQBar:
	if( std::abs(genEvt->daughterQuarkBarOfWPlus()->pdgId())==4 )
	  setCandidate(jets, match[idx], lightQBar_, jetCorrectionLevel("cQuark"));
	else
	  setCandidate(jets, match[idx], lightQBar_, jetCorrectionLevel("udsQuark"));
	break;
      case TtFullHadEvtPartons::B:
	setCandidate(jets, match[idx], b_          , jetCorrectionLevel("bQuark")); break;
      case TtFullHadEvtPartons::LightP:
	if( std::abs(genEvt->daughterQuarkOfWMinus()->pdgId())==4 )
	  setCandidate(jets, match[idx], lightP_   , jetCorrectionLevel("cQuark"));
	else
	  setCandidate(jets, match[idx], lightP_   , jetCorrectionLevel("udsQuark"));
	break;
      case TtFullHadEvtPartons::LightPBar:
	if( std::abs(genEvt->daughterQuarkBarOfWMinus()->pdgId())==4 )
	  setCandidate(jets, match[idx], lightPBar_, jetCorrectionLevel("cQuark"));
	else
	  setCandidate(jets, match[idx], lightPBar_, jetCorrectionLevel("udsQuark"));
	break;
      case TtFullHadEvtPartons::BBar:
	setCandidate(jets, match[idx], bBar_       , jetCorrectionLevel("bQuark")); break;	
      }
    }
  }
}
