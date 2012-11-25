#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <DataFormats/Candidate/interface/CompositeCandidate.h>
#include <DataFormats/Candidate/interface/CompositeCandidateFwd.h>
#include <DataFormats/Common/interface/RefToPtr.h>
#include "DataFormats/Common/interface/View.h"
#include <DataFormats/MuonReco/interface/MuonFwd.h>
#include "DataFormats/Common/interface/Handle.h"

#include <iostream>
#include <iomanip>

namespace
{
  bool higherPt(const reco::CandidateBaseRef& muon1, const reco::CandidateBaseRef& muon2)
  {
    return (muon1->pt() > muon2->pt());
  }
}

std::vector<reco::CandidateBaseRef> getSelMuons(edm::Event& evt, const edm::InputTag& srcSelMuons)
{
  std::vector<reco::CandidateBaseRef> selMuons;

  edm::Handle<reco::CompositeCandidateCollection> combCandidatesHandle;
  if ( evt.getByLabel(srcSelMuons, combCandidatesHandle) ) {
    if ( combCandidatesHandle->size() >= 1 ) {
      const reco::CompositeCandidate& combCandidate = combCandidatesHandle->at(0); // TF: use only the first combined candidate
      for ( size_t idx = 0; idx < combCandidate.numberOfDaughters(); ++idx ) { 
	const reco::Candidate* daughter = combCandidate.daughter(idx);
	reco::CandidateBaseRef selMuon;
	if ( daughter->hasMasterClone() ) {
	  selMuon = daughter->masterClone();
	} 
	if ( selMuon.isNull() ) 
	  throw cms::Exception("Configuration") 
	    << "Collection 'selectedMuons' = " << srcSelMuons.label() << " of CompositeCandidates does not refer to daughters of valid type !!\n";
	selMuons.push_back(selMuon);
      }
    }
  } else {
    typedef edm::View<reco::Candidate> CandidateView;
    edm::Handle<CandidateView> selMuonsHandle;
    if ( evt.getByLabel(srcSelMuons, selMuonsHandle) ) {
      for ( size_t idx = 0; idx < selMuonsHandle->size(); ++idx ) {
	selMuons.push_back(reco::CandidateBaseRef(selMuonsHandle->refAt(idx)));
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

reco::CandidateBaseRef getTheMuPlus(const std::vector<reco::CandidateBaseRef>& selMuons)
{
//--- return highest Pt muon of positive charge
//
//    NOTE: function assumes that collection of muons passed as function argument is sorted by decreasing Pt
//         (= as returned by 'getSelMuons' function)
  
  for ( std::vector<reco::CandidateBaseRef>::const_iterator selMuon = selMuons.begin();
	selMuon != selMuons.end(); ++selMuon ) {
    if ( (*selMuon)->charge() > +0.5 ) return (*selMuon);
  }

  // no muon of positive charge found
  return reco::CandidateBaseRef();
}

reco::CandidateBaseRef getTheMuMinus(const std::vector<reco::CandidateBaseRef>& selMuons)
{
//--- return highest Pt muon of negative charge
//
//    NOTE: function assumes that collection of muons passed as function argument is sorted by decreasing Pt
//         (= as returned by 'getSelMuons' function)

  for ( std::vector<reco::CandidateBaseRef>::const_iterator selMuon = selMuons.begin();
	selMuon != selMuons.end(); ++selMuon ) {
    if ( (*selMuon)->charge() < -0.5 ) return (*selMuon);
  }

  // no muon of negative charge found
  return reco::CandidateBaseRef();
}

TrackDetMatchInfo getTrackDetMatchInfo(const edm::Event& evt, const edm::EventSetup& es, 
				       TrackDetectorAssociator& trackAssociator, const TrackAssociatorParameters& trackAssociatorParameters, 
				       const reco::Candidate* muon)
{
  TrackDetMatchInfo trackDetMatchInfo;
  const reco::Muon* recoMuon = dynamic_cast<const reco::Muon*>(muon);
  if ( recoMuon && recoMuon->globalTrack().isNonnull() ) {
    trackDetMatchInfo = trackAssociator.associate(evt, es, *recoMuon->globalTrack(), trackAssociatorParameters);
  } else {
    GlobalVector muonP3(muon->px(), muon->py(), muon->pz()); 
    GlobalPoint muonVtx(muon->vertex().x(), muon->vertex().y(), muon->vertex().z());
    trackDetMatchInfo = trackAssociator.associate(evt, es, muonP3, muonVtx, muon->charge(), trackAssociatorParameters);
  }
  return trackDetMatchInfo;
}
