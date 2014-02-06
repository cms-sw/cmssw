
/** \class GlobalMuonTracksMixer
 *
 * Merge collections of reco::Tracks
 * of original Z -> mumu events (after removing the reconstructed muons)
 * and embedded tau decay products.
 * 
 * This class takes care of updating the reco::MuonTrackLinks objects
 *
 * \author Tomasz Maciej Frueboes;
 *         Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: GlobalMuonTrackMixer.cc,v 1.1 2013/03/30 16:41:12 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"

#include "TauAnalysis/MCEmbeddingTools/interface/TrackMixerBase.h"

#include <vector>

class GlobalMuonTrackMixer : public TrackMixerBase 
{
 public:
  explicit GlobalMuonTrackMixer(const edm::ParameterSet&);
  ~GlobalMuonTrackMixer() {}

 private:
  virtual void produceTrackExtras(edm::Event&, const edm::EventSetup&);

  typedef std::vector<reco::MuonTrackLinks> MuonTrackLinksCollection;
};

GlobalMuonTrackMixer::GlobalMuonTrackMixer(const edm::ParameterSet& cfg) 
  : TrackMixerBase(cfg)
{  
  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
     produces<MuonTrackLinksCollection>(todoItem->srcTrackCollection1_.instance());
  }
}

namespace
{
  reco::MuonTrackLinks makeMuonTrackLink(const reco::MuonTrackLinks& trackLink, const std::map<reco::TrackRef, reco::TrackRef>& trackRefMap)
  {
    reco::MuonTrackLinks trackLink_output;
    trackLink_output.setTrackerTrack(trackLink.trackerTrack());
    trackLink_output.setStandAloneTrack(trackLink.standAloneTrack());
    trackLink_output.setGlobalTrack(trackLink.globalTrack());
    
    // update edm::Refs to globalMuon collection
    for ( std::map<reco::TrackRef, reco::TrackRef>::const_iterator outputToInputTrackAssociation = trackRefMap.begin();
	  outputToInputTrackAssociation != trackRefMap.end(); ++outputToInputTrackAssociation ) {
      reco::TrackRef track_output = outputToInputTrackAssociation->first;
      reco::TrackRef track_input = outputToInputTrackAssociation->second;
      if ( track_input == trackLink.globalTrack() ) trackLink_output.setGlobalTrack(track_output);
    }
    
    return trackLink_output;
  }
}

void GlobalMuonTrackMixer::produceTrackExtras(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) std::cout << "<GlobalMuonTrackMixer::produceTrackExtras (" << moduleLabel_ << ")>:" << std::endl;

  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
    edm::Handle<MuonTrackLinksCollection> trackLinksCollection1;
    evt.getByLabel(todoItem->srcTrackCollection1_, trackLinksCollection1);

    edm::Handle<MuonTrackLinksCollection> trackLinksCollection2;
    evt.getByLabel(todoItem->srcTrackCollection2_, trackLinksCollection2);

    std::auto_ptr<MuonTrackLinksCollection> trackLinks_output(new MuonTrackLinksCollection());

    for ( MuonTrackLinksCollection::const_iterator trackLink = trackLinksCollection1->begin();
	  trackLink != trackLinksCollection1->end(); ++trackLink ) {
      reco::MuonTrackLinks trackLink_output =  makeMuonTrackLink(*trackLink, todoItem->trackRefMap_);
      trackLinks_output->push_back(trackLink_output);
    }
    for ( MuonTrackLinksCollection::const_iterator trackLink = trackLinksCollection2->begin();
	  trackLink != trackLinksCollection2->end(); ++trackLink ) {
      reco::MuonTrackLinks trackLink_output =  makeMuonTrackLink(*trackLink, todoItem->trackRefMap_);
      trackLinks_output->push_back(trackLink_output);
    }

    if ( verbosity_ ) {
      std::cout << "instanceLabel = " << todoItem->srcTrackCollection1_.instance() << ": #entries = " << trackLinks_output->size() << std::endl;
      int idx = 0;
      for ( MuonTrackLinksCollection::const_iterator trackLink = trackLinks_output->begin();
	    trackLink != trackLinks_output->end(); ++trackLink ) {
	std::cout << "trackLink #" << idx << ":" << std::endl;
	std::cout << " trackerTrack = " << trackLink->trackerTrack().id() << ":" << trackLink->trackerTrack().key() << std::endl;
	std::cout << " standAloneTrack = " << trackLink->standAloneTrack().id() << ":" << trackLink->standAloneTrack().key() << std::endl;
	std::cout << " globalTrack = " << trackLink->globalTrack().id() << ":" << trackLink->globalTrack().key() << std::endl;
	++idx;
      }
    }

    evt.put(trackLinks_output, todoItem->srcTrackCollection1_.instance());
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GlobalMuonTrackMixer);
