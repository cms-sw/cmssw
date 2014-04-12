#ifndef TauAnalysis_MCEmbeddingTools_TrackMixerBase_h
#define TauAnalysis_MCEmbeddingTools_TrackMixerBase_h

/** \class TrackMixerBase
 *
 * Base class for modules merging collections of reco::Tracks
 * of original Z -> mumu events (after removing the reconstructed muons)
 * and embedded tau decay products.
 * 
 * \author Tomasz Maciej Frueboes;
 *         Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: TrackMixerBase.h,v 1.1 2013/03/29 15:55:18 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>

class TrackMixerBase : public edm::EDProducer 
{
 public:
  explicit TrackMixerBase(const edm::ParameterSet&);
  ~TrackMixerBase() {}

 protected:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  virtual void produceTracks(edm::Event&, const edm::EventSetup&);
  virtual void produceTrackExtras(edm::Event&, const edm::EventSetup&) = 0;

  std::string moduleLabel_;

  struct todoListEntryType
  {
    edm::InputTag srcTrackCollection1_;
    edm::InputTag srcTrackCollection2_;

    mutable std::map<reco::TrackRef, reco::TrackRef> trackRefMap_; // key = edm::Ref to output track collection, value = edm::Ref to input track collections
                                                                   // (needed by TeVMuonTrackMixer)
  };
  std::vector<todoListEntryType> todoList_; 

  int verbosity_;
};

#endif

