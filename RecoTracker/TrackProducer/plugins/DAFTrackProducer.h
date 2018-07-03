/** \class DAFTrackProducer
  *  EDProducer for DAFTrackProducerAlgorithm.
  *
  *  \author tropiano, genta
  *  \review in May 2014 by brondolin 
  */

#ifndef DAFTrackProducer_h
#define DAFTrackProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoTracker/TrackProducer/interface/KfTrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/DAFTrackProducerAlgorithm.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajAnnealing.h"

class DAFTrackProducer : public KfTrackProducerBase, public edm::stream::EDProducer<> {
public:

  typedef std::vector<Trajectory> TrajectoryCollection;
//  typedef std::vector<TrajAnnealing> TrajAnnealingCollection;
  explicit DAFTrackProducer(const edm::ParameterSet& iConfig);

  // Implementation of produce method
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  DAFTrackProducerAlgorithm theAlgo;
  using TrackProducerBase<reco::Track>::getFromEvt;
  void getFromEvt(edm::Event&, edm::Handle<TrajTrackAssociationCollection>&, reco::BeamSpot&);
  void putInEvtTrajAnn(edm::Event& theEvent, TrajAnnealingCollection & trajannResults,
                std::unique_ptr<TrajAnnealingCollection>& selTrajAnn);

  bool TrajAnnSaving_;
  edm::EDGetToken srcTT_;
  
};

#endif
