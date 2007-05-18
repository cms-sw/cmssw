#ifndef TrackProducer_h
#define TrackProducer_h

/** \class TrackProducer
 *  Produce Tracks from TrackCandidates
 *
 *  $Date: 2007/03/26 10:13:49 $
 *  $Revision: 1.1 $
 *  \author cerati
 */

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class TrackProducer : public TrackProducerBase, public edm::EDProducer {
public:

  /// Constructor
  explicit TrackProducer(const edm::ParameterSet& iConfig);

  /// Implementation of produce method
  virtual void produce(edm::Event&, const edm::EventSetup&);

  /// Get Transient Tracks
  std::vector<reco::TransientTrack> getTransient(edm::Event&, const edm::EventSetup&);

private:
  TrackProducerAlgorithm theAlgo;

};

#endif
