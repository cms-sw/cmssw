#ifndef TrackRefitter_h
#define TrackRefitter_h

/** \class TrackRefitter
 *  Refit Tracks: Produce Tracks from TrackCollection. It performs a new final fit on a TrackCollection.
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

class TrackRefitter : public KfTrackProducerBase, public edm::stream::EDProducer<> {
public:

  /// Constructor
  explicit TrackRefitter(const edm::ParameterSet& iConfig);

  /// Implementation of produce method
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:
  TrackProducerAlgorithm<reco::Track> theAlgo;
  enum Constraint { none, momentum, vertex, trackParameters };
  Constraint constraint_;
  edm::EDGetToken trkconstrcoll_;

};

#endif
