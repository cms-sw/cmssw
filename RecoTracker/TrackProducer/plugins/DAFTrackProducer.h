#ifndef DAFTrackProducer_h
#define DAFTrackProducer_h

#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/DAFTrackProducerAlgorithm.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class DAFTrackProducer : public KfTrackProducerBase, public edm::EDProducer {
public:
  typedef std::vector<Trajectory> TrajectoryCollection;
  /// Constructor
  explicit DAFTrackProducer(const edm::ParameterSet& iConfig);

  /// Implementation of produce method
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  DAFTrackProducerAlgorithm theAlgo;
  void getFromEvt(edm::Event&, edm::Handle<TrajectoryCollection>&, reco::BeamSpot&);
};

#endif
