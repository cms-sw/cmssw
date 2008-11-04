#ifndef MTFTrackProducer_h
#define MTFTrackProducer_h

#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/MTFTrackProducerAlgorithm.h"

class MTFTrackProducer : public KfTrackProducerBase, public edm::EDProducer {
public:

  /// Constructor
  explicit MTFTrackProducer(const edm::ParameterSet& iConfig);

  /// Implementation of produce method
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  MTFTrackProducerAlgorithm theAlgo;

};

#endif
