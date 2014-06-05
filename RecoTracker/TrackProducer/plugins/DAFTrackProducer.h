/** \class DAFTrackProducer
  *  EDProducer for DAFTrackProducerAlgorithm.
  *
  *  \author tropiano, genta
  *  \review in May 2014 by brondolin 
  */

#ifndef DAFTrackProducer_h
#define DAFTrackProducer_h

#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/DAFTrackProducerAlgorithm.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajAnnealing.h"

class DAFTrackProducer : public KfTrackProducerBase, public edm::EDProducer {
public:

  typedef std::vector<Trajectory> TrajectoryCollection;
//  typedef std::vector<TrajAnnealing> TrajAnnealingCollection;
  explicit DAFTrackProducer(const edm::ParameterSet& iConfig);

  // Implementation of produce method
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  DAFTrackProducerAlgorithm theAlgo;
  void getFromEvt(edm::Event&, edm::Handle<TrajectoryCollection>&, reco::BeamSpot&);
  void putInEvtTrajAnn(edm::Event& theEvent, TrajAnnealingCollection & trajannResults,
                std::auto_ptr<TrajAnnealingCollection>& selTrajAnn);
};

#endif
