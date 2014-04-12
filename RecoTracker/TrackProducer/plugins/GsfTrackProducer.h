#ifndef GsfTrackProducer_h
#define GsfTrackProducer_h

#include "RecoTracker/TrackProducer/interface/GsfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfComponent5D.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"

class GsfTrackProducer : public GsfTrackProducerBase, public edm::EDProducer {
public:

  explicit GsfTrackProducer(const edm::ParameterSet& iConfig);


  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:
  TrackProducerAlgorithm<reco::GsfTrack> theAlgo;

};

#endif
