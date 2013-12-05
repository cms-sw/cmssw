#include "TauAnalysis/MCEmbeddingTools/interface/TrackMixerBase.h"

TrackMixerBase::TrackMixerBase(const edm::ParameterSet& cfg) 
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  edm::VParameterSet todoList = cfg.getParameter<edm::VParameterSet>("todo");
  if ( todoList.size() == 0 ) {
    throw cms::Exception("Configuration") 
      << "Empty to-do list !!\n";
  }
  
  for ( edm::VParameterSet::const_iterator todoItem = todoList.begin();
	todoItem != todoList.end(); ++todoItem ) {
    todoListEntryType todoListEntry;
    todoListEntry.srcTrackCollection1_ = todoItem->getParameter<edm::InputTag>("collection1");
    todoListEntry.srcTrackCollection2_ = todoItem->getParameter<edm::InputTag>("collection2");
    
    std::string instanceLabel1 = todoListEntry.srcTrackCollection1_.instance();
    std::string instanceLabel2 = todoListEntry.srcTrackCollection2_.instance();
    if ( instanceLabel1 != instanceLabel2 ) {
      throw cms::Exception("Configuration") 
	<< "Mismatch in Instance labels for collection 1 = " << instanceLabel1 << " and 2 = " << instanceLabel2 << " !!\n";
    }
    
    todoList_.push_back(todoListEntry); 

    produces<reco::TrackCollection>(instanceLabel1);
  }

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

void TrackMixerBase::produce(edm::Event& evt, const edm::EventSetup& es)
{
  produceTracks(evt, es);
  produceTrackExtras(evt, es);
}

void TrackMixerBase::produceTracks(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) std::cout << "<TrackMixerBase::produce (" << moduleLabel_ << ")>:" << std::endl;

  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
    todoItem->trackRefMap_.clear();

    edm::Handle<reco::TrackCollection> trackCollection1;
    evt.getByLabel(todoItem->srcTrackCollection1_, trackCollection1);

    edm::Handle<reco::TrackCollection> trackCollection2;
    evt.getByLabel(todoItem->srcTrackCollection2_, trackCollection2);
    
    if ( verbosity_ ) {
      std::cout << "trackCollection(input1 = " << todoItem->srcTrackCollection1_.label() << ":" << todoItem->srcTrackCollection1_.instance() << ":" << todoItem->srcTrackCollection1_.process() << "):" 
		<< " #entries = " << trackCollection1->size() << std::endl;
      std::cout << "trackCollection(input2 = " << todoItem->srcTrackCollection2_.label() << ":" << todoItem->srcTrackCollection2_.instance() << ":" << todoItem->srcTrackCollection2_.process() << "):" 
		<< " #entries = " << trackCollection2->size() << std::endl;
    }

    std::auto_ptr<reco::TrackCollection> trackCollection_output(new reco::TrackCollection());

    reco::TrackRefProd trackCollectionRefProd_output = evt.getRefBeforePut<reco::TrackCollection>(todoItem->srcTrackCollection1_.instance());
    size_t idxTrack_output = 0;

    size_t numTracks1 = trackCollection1->size();
    for ( size_t idxTrack1 = 0; idxTrack1 < numTracks1; ++idxTrack1 ) {
      reco::TrackRef track1(trackCollection1, idxTrack1);      
      trackCollection_output->push_back(*track1);
      todoItem->trackRefMap_[reco::TrackRef(trackCollectionRefProd_output, idxTrack_output)] = track1;
      ++idxTrack_output;
    }
    
    size_t numTracks2 = trackCollection2->size();
    for ( size_t idxTrack2 = 0; idxTrack2 < numTracks2; ++idxTrack2 ) {
      reco::TrackRef track2(trackCollection2, idxTrack2);
      trackCollection_output->push_back(*track2);
      todoItem->trackRefMap_[reco::TrackRef(trackCollectionRefProd_output, idxTrack_output)] = track2;
      ++idxTrack_output;
    }

    if ( verbosity_ ) {      
      std::cout << "trackCollection(output = " << moduleLabel_ << ":" << todoItem->srcTrackCollection1_.instance() << "): #entries = " << trackCollection_output->size() << std::endl;
      int idx = 0;
      for ( reco::TrackCollection::const_iterator track = trackCollection_output->begin();
	    track != trackCollection_output->end(); ++track ) {
	if ( track->pt() > 5. ) {
	  std::cout << "track #" << idx << ": Pt = " << track->pt() << ", eta = " << track->eta() << ", phi = " << track->phi() << std::endl;
	  ++idx;
	}
      }
    }  

    evt.put(trackCollection_output, todoItem->srcTrackCollection1_.instance());
  }
}
