#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiHypothesisGenMatch.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiEvtPartons.h"
#include "DataFormats/Math/interface/deltaR.h"

TtSemiHypothesisGenMatch::TtSemiHypothesisGenMatch(const edm::ParameterSet& cfg):
  TtSemiHypothesis( cfg ) { }

TtSemiHypothesisGenMatch::~TtSemiHypothesisGenMatch() { }

void
TtSemiHypothesisGenMatch::buildHypo(edm::Event& evt,
				    const edm::Handle<edm::View<reco::RecoCandidate> >& leps, 
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
      case TtSemiEvtPartons::LightQ:
	{ 
	  edm::Ptr<pat::Jet> jet = edm::Ptr<pat::Jet>(jets, ij);
	  lightQ_   = new reco::ShallowClonePtrCandidate( jet, jet->charge(), jet->p4(), jet->vertex() ); 
	  break;
	}
      case TtSemiEvtPartons::LightQBar:
	{
	  edm::Ptr<pat::Jet> jet = edm::Ptr<pat::Jet>(jets, ij);
	  lightQBar_= new reco::ShallowClonePtrCandidate( jet, jet->charge(), jet->p4(), jet->vertex() ); 
	  break;
	}
      case TtSemiEvtPartons::HadB:
	{
	  edm::Ptr<pat::Jet> jet = edm::Ptr<pat::Jet>(jets, ij);
	  hadronicB_= new reco::ShallowClonePtrCandidate( jet, jet->charge(), jet->p4(), jet->vertex() ); 
	  break;
	}
      case TtSemiEvtPartons::LepB:
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
    int iLepton = findMatchingLepton(evt,leps);
    if( iLepton>=0 ){
      edm::Ptr<reco::RecoCandidate> lep = edm::Ptr<reco::RecoCandidate>(leps, iLepton);
      lepton_= new reco::ShallowClonePtrCandidate( lep, lep->charge(), lep->p4(), lep->vertex() );
    }
  }

  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  if( !mets->empty() ){
    edm::Ptr<pat::MET> met = edm::Ptr<pat::MET>(mets, 0);
    neutrino_= new reco::ShallowClonePtrCandidate( met, met->charge(), met->p4(), met->vertex() );
  }
}

int
TtSemiHypothesisGenMatch::findMatchingLepton(edm::Event& evt, const edm::Handle<edm::View<reco::RecoCandidate> >& leps)
{
  int genIdx=-1;

  // set genEvent
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);  
  
  if( genEvt->isTtBar() && genEvt->isSemiLeptonic() && genEvt->lepton() ){
    double minDR=-1;
    for(unsigned i=0; i<leps->size(); ++i){
      double dR = deltaR(genEvt->lepton()->eta(), genEvt->lepton()->phi(), (*leps)[i].eta(), (*leps)[i].phi());
      if(minDR<0 || dR<minDR){
	minDR=dR;
	genIdx=i;
      }
    }
  }
  return genIdx;
}
