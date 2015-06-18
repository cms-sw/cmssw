
/** \class TeVMuonTrackCleaner
 *
 * Produce collection of reco::Tracks in Z --> mu+ mu- event
 * from which the two muons are removed 
 * (later to be replaced by simulated tau decay products) 
 * 
 * This class takes care of updating the edm::AssociationMap
 * between globalMuons and tevMuons
 *
 * \authors Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: TeVMuonTrackCleaner.cc,v 1.1 2013/03/30 16:41:12 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/MuonTrackCleanerBase.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <vector>
#include <algorithm>

typedef edm::AssociationMap<edm::OneToOne<reco::TrackCollection, reco::TrackCollection> > TrackToTrackMap;

class TeVMuonTrackCleaner : public MuonTrackCleanerBase
{
 public:
  explicit TeVMuonTrackCleaner(const edm::ParameterSet&);
  ~TeVMuonTrackCleaner() {}

 private:
  virtual void produceTrackExtras(edm::Event&, const edm::EventSetup&);

  edm::InputTag srcGlobalMuons_cleaned_;
};

TeVMuonTrackCleaner::TeVMuonTrackCleaner(const edm::ParameterSet& cfg)
  : MuonTrackCleanerBase(cfg),
    srcGlobalMuons_cleaned_(cfg.getParameter<edm::InputTag>("srcGlobalMuons_cleaned"))
{
  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
    produces<TrackToTrackMap>(todoItem->srcTracks_.instance());
  }
}

namespace
{
  void matchMuonTracks(const reco::TrackRef& globalMuonTrack_cleaned, const TrackToTrackMap& trackToTrackMap, bool& isMatched, reco::TrackRef& tevMuonTrack_matched, double& dRmatch, int verbosity)
  {
    for ( typename TrackToTrackMap::const_iterator entry = trackToTrackMap.begin();
	  entry != trackToTrackMap.end(); ++entry ) {
      reco::TrackRef globalMuonTrack_uncleaned = entry->key;
      double dR = reco::deltaR(globalMuonTrack_uncleaned->eta(), globalMuonTrack_uncleaned->phi(), globalMuonTrack_cleaned->eta(), globalMuonTrack_cleaned->phi());
      if ( verbosity ) {
	std::cout << "globalMuon(uncleaned = " << globalMuonTrack_uncleaned.id() << ":" << globalMuonTrack_uncleaned.key() << "):" 
		  << " Pt = " << globalMuonTrack_uncleaned->pt() << ", eta = " << globalMuonTrack_uncleaned->eta() << ", phi = " << globalMuonTrack_uncleaned->phi() << ","
		  << " dR = " << dR << std::endl;
      }
      if ( dR < 1.e-2 && dR < dRmatch ) {	  
	isMatched = true;
	tevMuonTrack_matched = entry->val;
	dRmatch = dR;
      }
    }
  }
}

void TeVMuonTrackCleaner::produceTrackExtras(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) std::cout << "<TeVMuonTrackCleaner::produceTrackExtras (" << moduleLabel_ << ")>:" << std::endl;

  edm::Handle<reco::TrackCollection> globalMuons_cleaned;
  evt.getByLabel(srcGlobalMuons_cleaned_, globalMuons_cleaned);
  if ( verbosity_ ) {
    std::cout << "globalMuons(cleaned = " << srcGlobalMuons_cleaned_.label() << ":" << srcGlobalMuons_cleaned_.instance() << ":" << srcGlobalMuons_cleaned_.process() << "," 
	      << " productId = " << globalMuons_cleaned.id() << "): #entries = " << globalMuons_cleaned->size() << std::endl;
  }

  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
    
    edm::Handle<TrackToTrackMap> trackToTrackMap;
    evt.getByLabel(todoItem->srcTracks_, trackToTrackMap);
    if ( verbosity_ ) {
      for ( typename TrackToTrackMap::const_iterator entry = trackToTrackMap->begin();
	    entry != trackToTrackMap->end(); ++entry ) {
	std::cout << "trackToTrackMap[" << entry->key.id() << ":" << entry->key.key() << "] = " << entry->val.id() << ":" << entry->val.key() << std::endl;
      }
    }

    std::auto_ptr<TrackToTrackMap> trackToTrackMap_cleaned(new TrackToTrackMap(globalMuons_cleaned, trackToTrackMap->refProd().val));
    
    size_t numGlobalMuons_cleaned = globalMuons_cleaned->size();
    for ( size_t iGlobalMuons_cleaned = 0; iGlobalMuons_cleaned < numGlobalMuons_cleaned; ++iGlobalMuons_cleaned ) {
      reco::TrackRef globalMuonTrack_cleaned(globalMuons_cleaned, iGlobalMuons_cleaned);
      if ( verbosity_ ) {
	std::cout << " globalMuon(cleaned = " << globalMuonTrack_cleaned.id() << ":" << globalMuonTrack_cleaned.key() << "):" 
		  << " Pt = " << globalMuonTrack_cleaned->pt() << ", eta = " << globalMuonTrack_cleaned->eta() << ", phi = " << globalMuonTrack_cleaned->phi() << std::endl;
      }
      bool isMatched = false;
      reco::TrackRef tevMuonTrack_matched;
      double dRmatch = 1.e+3;   
      matchMuonTracks(globalMuonTrack_cleaned, *trackToTrackMap, isMatched, tevMuonTrack_matched, dRmatch, verbosity_);
      if ( isMatched ) {
	if ( verbosity_ ) {
	  std::cout << "--> adding trackToTrackMap[" << globalMuonTrack_cleaned.id() << ":" << globalMuonTrack_cleaned.key() << "]" 
		    << " = " << tevMuonTrack_matched.id() << ":" << tevMuonTrack_matched.key() << std::endl;
	}
	trackToTrackMap_cleaned->insert(globalMuonTrack_cleaned, tevMuonTrack_matched);
      } else {
	throw cms::Exception("TeVMuonTrackCleaner::produceTrackExtras") 
	  << "Failed to find Track association for " << globalMuonTrack_cleaned.id() << ":" << globalMuonTrack_cleaned.key() << "!!\n";	
      }
    }
    
    evt.put(trackToTrackMap_cleaned, todoItem->srcTracks_.instance());
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TeVMuonTrackCleaner);
