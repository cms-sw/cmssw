#ifndef GsfTrackProducer_h
#define GsfTrackProducer_h

#include "RecoTracker/TrackProducer/interface/GsfTrackProducerBase.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class GsfTrackProducer : public GsfTrackProducerBase, public edm::EDProducer {
public:

  explicit GsfTrackProducer(const edm::ParameterSet& iConfig);


  virtual void produce(edm::Event&, const edm::EventSetup&);

//   std::vector<reco::TransientTrack> getTransient(edm::Event&, const edm::EventSetup&);

private:
  GsfTrackProducerAlgorithm theAlgo;

};

#endif
