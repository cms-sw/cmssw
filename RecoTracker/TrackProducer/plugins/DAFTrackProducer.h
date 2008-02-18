#ifndef DAFTrackProducer_h
#define DAFTrackProducer_h

#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/DAFTrackProducerAlgorithm.h"

class DAFTrackProducer : public KfTrackProducerBase, public edm::EDProducer {
public:

  /// Constructor
  explicit DAFTrackProducer(const edm::ParameterSet& iConfig);

  /// Implementation of produce method
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  DAFTrackProducerAlgorithm theAlgo;

};

#endif
