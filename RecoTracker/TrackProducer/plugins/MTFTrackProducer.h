#ifndef MTFTrackProducer_h
#define MTFTrackProducer_h

#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/MTFTrackProducerAlgorithm.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class MTFTrackProducer : public KfTrackProducerBase, public edm::EDProducer {
public:
  typedef std::vector<Trajectory> TrajectoryCollection;
  /// Constructor
  explicit MTFTrackProducer(const edm::ParameterSet& iConfig);

  /// Implementation of produce method
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  MTFTrackProducerAlgorithm theAlgo;
  void getFromEvt(edm::Event&, edm::Handle<TrajectoryCollection>&, reco::BeamSpot&);
};

#endif
