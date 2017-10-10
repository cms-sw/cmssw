#ifndef TrackProducer_h
#define TrackProducer_h

/** \class TrackProducer
 *  Produce Tracks from TrackCandidates
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class TrackProducer : public KfTrackProducerBase, public edm::stream::EDProducer<> {
public:

  /// Constructor
  explicit TrackProducer(const edm::ParameterSet& iConfig);

  /// Implementation of produce method
  void produce(edm::Event&, const edm::EventSetup&) override;

  /// Get Transient Tracks
  std::vector<reco::TransientTrack> getTransient(edm::Event&, const edm::EventSetup&);

//   /// Put produced collections in the event
//   virtual void putInEvt(edm::Event&,
// 			std::unique_ptr<TrackingRecHitCollection>&,
// 			std::unique_ptr<TrackCollection>&,
// 			std::unique_ptr<reco::TrackExtraCollection>&,
// 			std::unique_ptr<std::vector<Trajectory> >&,
// 			AlgoProductCollection&);

private:
  TrackProducerAlgorithm<reco::Track> theAlgo;

};

#endif
