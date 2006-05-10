#ifndef TrackRefitter_h
#define TrackRefitter_h

//
// Package:    RecoTracker/TrackProducer
// Class:      TrackRefitter
// 
//
// Description: Refit Tracks
//
//
// Original Author:  Giuseppe Cerati
//         Created:  Wed May  10 14:29:31 CET 2006
//

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

class TrackRefitter : public TrackProducerBase, public edm::EDProducer {
public:

  explicit TrackRefitter(const edm::ParameterSet& iConfig);

  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  TrackProducerAlgorithm theAlgo;

};

#endif
