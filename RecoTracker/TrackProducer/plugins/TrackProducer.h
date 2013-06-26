#ifndef TrackProducer_h
#define TrackProducer_h

/** \class TrackProducer
 *  Produce Tracks from TrackCandidates
 *
 *  $Date: 2013/02/27 13:28:55 $
 *  $Revision: 1.2 $
 *  \author cerati
 */

#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class TrackProducer : public KfTrackProducerBase, public edm::EDProducer {
public:

  /// Constructor
  explicit TrackProducer(const edm::ParameterSet& iConfig);

  /// Implementation of produce method
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  /// Get Transient Tracks
  std::vector<reco::TransientTrack> getTransient(edm::Event&, const edm::EventSetup&);

//   /// Put produced collections in the event
//   virtual void putInEvt(edm::Event&,
// 			std::auto_ptr<TrackingRecHitCollection>&,
// 			std::auto_ptr<TrackCollection>&,
// 			std::auto_ptr<reco::TrackExtraCollection>&,
// 			std::auto_ptr<std::vector<Trajectory> >&,
// 			AlgoProductCollection&);

private:
  TrackProducerAlgorithm<reco::Track> theAlgo;

};

#endif
