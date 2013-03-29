
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
 * \version $Revision: 1.5 $
 *
 * $Id: TeVMuonTrackMixer.cc,v 1.5 2012/10/14 12:22:24 veelken Exp $
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
  : TrackMixerBase(cfg)
{  
  for ( typename std::vector<todoListEntryType>::const_iterator todoItem = todoList_.begin();
	todoItem != todoList_.end(); ++todoItem ) {
     produces<TrackToTrackMap>(todoItem->srcTrackCollection1_.instance());
  }
}

namespace
{
  void matchMuonTracks(const reco::TrackRef& tevMuonTrack, const TrackToTrackMap& trackToTrackMap, reco::TrackRef& matchedTrack, int& numMatchesByRef, double& dRmatch, int verbosity)
  {
    for ( TrackToTrackMap::const_iterator trackToTrackAssociation = trackToTrackMap.begin();
	  trackToTrackAssociation != trackToTrackMap.end(); ++trackToTrackAssociation ) {
      reco::TrackRef globalMuonTrackRef = trackToTrackAssociation->key;
      reco::TrackRef tevMuonTrackRef = trackToTrackAssociation->val;
      if ( verbosity ) std::cout << "trackToTrackMap[" << globalMuonTrackRef.id() << ":" << globalMuonTrackRef.key() << "] = " << tevMuonTrackRef.id() << ":" << tevMuonTrackRef.key() << std::endl;
      if ( tevMuonTrackRef == tevMuonTrack ) {
	matchedTrack = globalMuonTrackRef;
	++numMatchesByRef;
      } else if ( numMatchesByRef == 0 ){
	double dR = reco::deltaR(tevMuonTrackRef->eta(), tevMuonTrackRef->phi(), tevMuonTrack->eta(), tevMuonTrack->phi());
	if ( dR < 1.e-2 && dR < dRmatch ) {
	  matchedTrack = globalMuonTrackRef;
	  dRmatch = dR;
	}
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
    edm::Handle<TrackToTrackMap> trackToTrackMapCollection1;
    evt.getByLabel(todoItem->srcTrackCollection1_, trackToTrackMapCollection1);

    edm::Handle<reco::TrackCollection> trackCollection2;
    evt.getByLabel(todoItem->srcTrackCollection2_, trackCollection2);    
    edm::Handle<TrackToTrackMap> trackToTrackMapCollection2;
    evt.getByLabel(todoItem->srcTrackCollection2_, trackToTrackMapCollection2);

    if ( verbosity_ ) {
      std::cout << "input1 (" << todoItem->srcTrackCollection1_.label() << ":" << todoItem->srcTrackCollection1_.instance() << ":" << todoItem->srcTrackCollection1_.process() << "):" << std::endl;
      std::cout << " trackCollection(productId = " << trackCollection1.id() << "): #entries = " << trackCollection1->size() << std::endl;
      std::cout << " trackToTrackCollectionMap(productId = " << trackToTrackMapCollection1.id() << "): #entries = " << trackToTrackMapCollection1->size() << std::endl;
      std::cout << "input2 (" << todoItem->srcTrackCollection2_.label() << ":" << todoItem->srcTrackCollection2_.instance() << ":" << todoItem->srcTrackCollection2_.process() << "):" << std::endl;
      std::cout << " trackCollection(productId = " << trackCollection2.id() << "): #entries = " << trackCollection2->size() << std::endl;
      std::cout << " trackToTrackCollectionMap(productId = " << trackToTrackMapCollection2.id() << "): #entries = " << trackToTrackMapCollection2->size() << std::endl;
    }

    std::auto_ptr<TrackToTrackMap> trackToTrackMap_output(new TrackToTrackMap());

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
      


    int idx = 0;
    for ( std::map<reco::TrackRef, reco::TrackRef>::const_iterator outputToInputTrackAssociation = todoItem->trackRefMap_.begin();
	  outputToInputTrackAssociation != todoItem->trackRefMap_.end(); ++outputToInputTrackAssociation ) {
      reco::TrackRef tevMuonTrack_output = outputToInputTrackAssociation->first;
      reco::TrackRef tevMuonTrack_input = outputToInputTrackAssociation->second;
      if ( verbosity_ ) std::cout << "trackRefMap[" << tevMuonTrack_input.id() << ":" << tevMuonTrack_input.key() << "] = " << tevMuonTrack_output.id() << ":" << tevMuonTrack_output.key() << std::endl;

      const reco::Track* globalMuonTrack_matched = 0;
      int numMatchesByRef = 0;
      double dRmatch = 1.e+3; 
      matchMuonTracks(tevMuonTrack_input, *trackToTrackMapCollection1, globalMuonTrack_matched, numMatchesByRef, dRmatch, verbosity_);
      matchMuonTracks(tevMuonTrack_input, *trackToTrackMapCollection2, globalMuonTrack_matched, numMatchesByRef, dRmatch, verbosity_);      
      if ( numMatchesByRef == 1 || (numMatchesByRef == 0 && dRmatch < 1.) ) {
	

	if ( verbosity_ ) {
	  std::cout << "--> adding trackToTrackMap[" << globalMuonTrack_matched.id() << ":" << globalMuonTrack_matched.key() << "]" 
		    << " = " << tevMuonTrack_output.id() << ":" << tevMuonTrack_output.key() << std::endl;
	}
	trackToTrackMap_output->insert(globalMuonTrack_matched, tevMuonTrack_output);
      } else throw cms::Exception("TeVMuonTrackMixer::produceTrackExtras") 
	  << "Failed to find unique Track association for " << tevMuonTrack_output.id() << ":" << tevMuonTrack_output.key() << "!!\n";
      
      ++idx;
    }

    evt.put(trackToTrackMap_output, todoItem->srcTrackCollection1_.instance());
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TeVMuonTrackMixer);
