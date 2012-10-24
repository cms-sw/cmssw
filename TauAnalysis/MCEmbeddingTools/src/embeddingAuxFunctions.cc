#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <DataFormats/Candidate/interface/CompositeCandidate.h>
#include <DataFormats/Candidate/interface/CompositeCandidateFwd.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>
#include "DataFormats/Common/interface/Handle.h"

namespace
{
  bool higherPt(const reco::Muon* muon1, const reco::Muon* muon2)
  {
    return (muon1->pt() > muon2->pt());
  }
}

std::vector<const reco::Muon*> getSelMuons(edm::Event& evt, const edm::InputTag& srcSelMuons)
{
  std::vector<const reco::Muon*> selMuons;

  edm::Handle<reco::CompositeCandidateCollection> combCandidatesHandle;
  if ( evt.getByLabel(srcSelMuons, combCandidatesHandle) ) {
    if ( combCandidatesHandle->size() >= 1 ) {
      const reco::CompositeCandidate& combCandidate = combCandidatesHandle->at(0); // TF: use only the first combined candidate
      for ( size_t idx = 0; idx < combCandidate.numberOfDaughters(); ++idx ) { 
	const reco::Candidate* daughter = combCandidate.daughter(idx);
	const reco::Muon* selMuon = 0;
	if ( daughter->hasMasterClone() ) {
	  selMuon = dynamic_cast<const reco::Muon*>(daughter->masterClone().get());
	} 
	if ( !selMuon ) 
	  throw cms::Exception("Configuration") 
	    << "Collection 'selectedMuons' = " << srcSelMuons.label() << " of CompositeCandidates does not refer to daughters of type reco::Muon !!\n";
	selMuons.push_back(selMuon);
      }
    }
  } else {
    edm::Handle<reco::MuonCollection> selMuonsHandle;
    if ( evt.getByLabel(srcSelMuons, selMuonsHandle) ) {
      for ( size_t idx = 0; idx < selMuonsHandle->size(); ++idx ) {
	selMuons.push_back(&selMuonsHandle->at(idx));
      }
    } else {
      throw cms::Exception("Configuration") 
	<< "Invalid input collection 'selectedMuons' = " << srcSelMuons.label() << " !!\n";
    }
  }

  // sort collection of selected muons by decreasing Pt
  std::sort(selMuons.begin(), selMuons.end(), higherPt);

  return selMuons;
}

const reco::Muon* getTheMuPlus(const std::vector<const reco::Muon*>& selMuons)
{
//--- return highest Pt muon of positive charge
//
//    NOTE: function assumes that collection of muons passed as function argument is sorted by decreasing Pt
//         (= as returned by 'getSelMuons' function)
  
  for ( std::vector<const reco::Muon*>::const_iterator selMuon = selMuons.begin();
	selMuon != selMuons.end(); ++selMuon ) {
    if ( (*selMuon)->charge() > +0.5 ) return (*selMuon);
  }

  // no muon of positive charge found
  return 0;
}

const reco::Muon* getTheMuMinus(const std::vector<const reco::Muon*>& selMuons)
{
//--- return highest Pt muon of negative charge
//
//    NOTE: function assumes that collection of muons passed as function argument is sorted by decreasing Pt
//         (= as returned by 'getSelMuons' function)

  for ( std::vector<const reco::Muon*>::const_iterator selMuon = selMuons.begin();
	selMuon != selMuons.end(); ++selMuon ) {
    if ( (*selMuon)->charge() > +0.5 ) return (*selMuon);
  }

  // no muon of negative charge found
  return 0;
}
