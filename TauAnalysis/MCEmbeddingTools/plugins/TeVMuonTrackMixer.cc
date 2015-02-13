
/** \class TeVMuonTrackMixer
 *
 * Merge collections of reco::Tracks
 * of original Z -> mumu events (after removing the reconstructed muons)
 * and embedded tau decay products.
 * 
 * This class takes care of updating the edm::AssociationMap
 * between globalMuons and tevMuons
 * 
 * \author Tomasz Maciej Frueboes;
 *         Christian Veelken, LLR
 *
 * \version $Revision: 1.2 $
 *
 * $Id: TeVMuonTrackMixer.cc,v 1.2 2013/03/30 13:22:48 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/TrackMixerBase.h"

#include <vector>

typedef edm::AssociationMap<edm::OneToOne<reco::TrackCollection, reco::TrackCollection> > TrackToTrackMap;

class TeVMuonTrackMixer : public TrackMixerBase 
{
 public:
  explicit TeVMuonTrackMixer(const edm::ParameterSet&);
  ~TeVMuonTrackMixer() {}

 private:
  virtual void produceTrackExtras(edm::Event&, const edm::EventSetup&);

  edm::InputTag srcGlobalMuons_cleaned_;
};

TeVMuonTrackMixer::TeVMuonTrackMixer(const edm::ParameterSet& cfg) 
  : TrackMixerBase(cfg),
    srcGlobalMuons_cleaned_(cfg.getParameter<edm::InputTag>("srcGlobalMuons_cleaned"))
{  
  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
     produces<TrackToTrackMap>(todoItem->srcTrackCollection1_.instance());
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

void TeVMuonTrackMixer::produceTrackExtras(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) std::cout << "<TeVMuonTrackMixer::produceTrackExtras (" << moduleLabel_ << ")>:" << std::endl;

  edm::Handle<reco::TrackCollection> globalMuons_cleaned;
  evt.getByLabel(srcGlobalMuons_cleaned_, globalMuons_cleaned);

  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {    
    edm::Handle<reco::TrackCollection> trackCollection1;
    evt.getByLabel(todoItem->srcTrackCollection1_, trackCollection1);
    edm::Handle<TrackToTrackMap> trackToTrackMap1;
    evt.getByLabel(todoItem->srcTrackCollection1_, trackToTrackMap1);

    edm::Handle<reco::TrackCollection> trackCollection2;
    evt.getByLabel(todoItem->srcTrackCollection2_, trackCollection2);    
    edm::Handle<TrackToTrackMap> trackToTrackMap2;
    evt.getByLabel(todoItem->srcTrackCollection2_, trackToTrackMap2);

    if ( verbosity_ ) {
      std::cout << "input1 (" << todoItem->srcTrackCollection1_.label() << ":" << todoItem->srcTrackCollection1_.instance() << ":" << todoItem->srcTrackCollection1_.process() << "):" << std::endl;
      std::cout << " trackCollection(productId = " << trackCollection1.id() << "): #entries = " << trackCollection1->size() << std::endl;
      std::cout << " trackToTrackMap(productId = " << trackToTrackMap1.id() << "): #entries = " << trackToTrackMap1->size() << std::endl;
      for ( typename TrackToTrackMap::const_iterator entry = trackToTrackMap1->begin();
	    entry != trackToTrackMap1->end(); ++entry ) {
	std::cout << "  trackToTrackMap[" << entry->key.id() << ":" << entry->key.key() << "] = " << entry->val.id() << ":" << entry->val.key() << std::endl;
      }
      std::cout << "input2 (" << todoItem->srcTrackCollection2_.label() << ":" << todoItem->srcTrackCollection2_.instance() << ":" << todoItem->srcTrackCollection2_.process() << "):" << std::endl;
      std::cout << " trackCollection(productId = " << trackCollection2.id() << "): #entries = " << trackCollection2->size() << std::endl;
      std::cout << " trackToTrackMap(productId = " << trackToTrackMap2.id() << "): #entries = " << trackToTrackMap2->size() << std::endl;
      for ( typename TrackToTrackMap::const_iterator entry = trackToTrackMap2->begin();
	    entry != trackToTrackMap2->end(); ++entry ) {
	std::cout << "  trackToTrackMap[" << entry->key.id() << ":" << entry->key.key() << "] = " << entry->val.id() << ":" << entry->val.key() << std::endl;
      }
    }

    TrackToTrackMap::ref_type::value_type refProd = trackToTrackMap1->refProd().val;
    if(trackToTrackMap1->empty()) refProd = trackToTrackMap2->refProd().val;
    std::auto_ptr<TrackToTrackMap> trackToTrackMap_output(new TrackToTrackMap(globalMuons_cleaned, refProd));

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
      matchMuonTracks(globalMuonTrack_cleaned, *trackToTrackMap1, isMatched, tevMuonTrack_matched, dRmatch, verbosity_);
      matchMuonTracks(globalMuonTrack_cleaned, *trackToTrackMap2, isMatched, tevMuonTrack_matched, dRmatch, verbosity_);
      if ( isMatched ) {
	if ( verbosity_ ) {
	  std::cout << "--> adding trackToTrackMap[" << globalMuonTrack_cleaned.id() << ":" << globalMuonTrack_cleaned.key() << "]" 
		    << " = " << tevMuonTrack_matched.id() << ":" << tevMuonTrack_matched.key() << std::endl;
	}
	trackToTrackMap_output->insert(globalMuonTrack_cleaned, tevMuonTrack_matched);
      } else {
	throw cms::Exception("TeVMuonTrackMixer::produceTrackExtras") 
	  << "Failed to find Track association for " << globalMuonTrack_cleaned.id() << ":" << globalMuonTrack_cleaned.key() << "!!\n";	
      }
    }

    evt.put(trackToTrackMap_output, todoItem->srcTrackCollection1_.instance());
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TeVMuonTrackMixer);
