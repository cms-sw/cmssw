
/** \class TeVMuonTrackCleaner
 *
 * Produce collection of reco::Tracks in Z --> mu+ mu- event
 * from which the two muons are removed 
 * (later to be replaced by simulated tau decay products) 
 * 
 * This class takes care of updating the reco::MuonTrackLinks objects
 *
 * \authors Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: GlobalMuonTrackCleaner.cc,v 1.1 2013/03/30 16:41:12 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/MuonTrackCleanerBase.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <vector>
#include <algorithm>

class GlobalMuonTrackCleaner : public MuonTrackCleanerBase
{
 public:
  explicit GlobalMuonTrackCleaner(const edm::ParameterSet&);
  ~GlobalMuonTrackCleaner() {}

 private:
  virtual void produceTrackExtras(edm::Event&, const edm::EventSetup&);

  edm::InputTag srcMuons_;

  typedef std::vector<reco::MuonTrackLinks> MuonTrackLinksCollection;
};

GlobalMuonTrackCleaner::GlobalMuonTrackCleaner(const edm::ParameterSet& cfg)
  : MuonTrackCleanerBase(cfg),
    srcMuons_(cfg.getParameter<edm::InputTag>("srcMuons"))
{
  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
    produces<MuonTrackLinksCollection>(todoItem->srcTracks_.instance());
  }
}

void GlobalMuonTrackCleaner::produceTrackExtras(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) std::cout << "<GlobalMuonTrackCleaner::produceTrackExtras (" << moduleLabel_ << ")>:" << std::endl;

  edm::Handle<reco::MuonCollection> muons;
  evt.getByLabel(srcMuons_, muons);

  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
    std::auto_ptr<MuonTrackLinksCollection> trackLinks_cleaned(new MuonTrackLinksCollection());
    
    for ( std::map<reco::TrackRef, reco::TrackRef>::const_iterator cleanedToUncleanedTrackAssociation = todoItem->trackRefMap_.begin();
	  cleanedToUncleanedTrackAssociation != todoItem->trackRefMap_.end(); ++cleanedToUncleanedTrackAssociation ) {
      const reco::Muon* matchedMuon = 0;
      double dRmatch = 1.e+3;      
      for ( reco::MuonCollection::const_iterator muon = muons->begin();
	    muon != muons->end(); ++muon ) {
	if ( muon->globalTrack().isNull() ) continue;	
	if ( muon->globalTrack() == cleanedToUncleanedTrackAssociation->second ) { // match by edm::Ref
	  matchedMuon = &(*muon);
	  dRmatch = 0.;
	  break;
	} else { // match by dR
	  double dR = reco::deltaR(muon->globalTrack()->eta(), muon->globalTrack()->phi(), cleanedToUncleanedTrackAssociation->second->eta(), cleanedToUncleanedTrackAssociation->second->phi());
	  if ( dR < 1.e-2 && dR < dRmatch ) {	  
	    matchedMuon = &(*muon);
	    dRmatch = dR;
	  }
	}
      }	   
      if ( matchedMuon ) {
	reco::MuonTrackLinks trackLink;
	trackLink.setTrackerTrack(matchedMuon->innerTrack());
	trackLink.setStandAloneTrack(matchedMuon->outerTrack());
	trackLink.setGlobalTrack(cleanedToUncleanedTrackAssociation->first);
	if ( verbosity_ ) {
	  std::cout << "creating new MuonTrackLinks:" << std::endl;
	  std::cout << " innerTrack = " << trackLink.trackerTrack().id() << ":" << trackLink.trackerTrack().key() << std::endl;
	  std::cout << " outerTrack = " << trackLink.standAloneTrack().id() << ":" << trackLink.standAloneTrack().key() << std::endl;
	  std::cout << " globalTrack = " << trackLink.globalTrack().id() << ":" << trackLink.globalTrack().key() << std::endl;
	}
	trackLinks_cleaned->push_back(trackLink);
      }
    }
    
    evt.put(trackLinks_cleaned, todoItem->srcTracks_.instance());
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GlobalMuonTrackCleaner);
