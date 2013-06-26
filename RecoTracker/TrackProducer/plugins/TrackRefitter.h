#ifndef TrackRefitter_h
#define TrackRefitter_h

/** \class TrackRefitter
 *  Refit Tracks: Produce Tracks from TrackCollection. It performs a new final fit on a TrackCollection.
 *
 *  $Date: 2013/02/27 13:28:56 $
 *  $Revision: 1.5 $
 *  \author cerati
 */

#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

class TrackRefitter : public KfTrackProducerBase, public edm::EDProducer {
public:

  /// Constructor
  explicit TrackRefitter(const edm::ParameterSet& iConfig);

  /// Implementation of produce method
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:
  TrackProducerAlgorithm<reco::Track> theAlgo;
  enum Constraint { none, momentum, vertex, trackParameters };
  Constraint constraint_;
  edm::InputTag trkconstrcoll_;

};

#endif
