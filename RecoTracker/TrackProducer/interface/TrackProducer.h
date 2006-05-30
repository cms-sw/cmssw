#ifndef TrackProducer_h
#define TrackProducer_h

//
// Package:    RecoTracker/TrackProducer
// Class:      TrackProducer
// 
//
// Description: Produce Tracks from TrackCandidates
//
//
// Original Author:  Giuseppe Cerati
//         Created:  Thu Mar  9 17:29:31 CET 2006
// $Id: TrackProducer.h,v 1.4 2006/05/10 15:02:20 magni Exp $
//

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"



class TrackProducer : public TrackProducerBase, public edm::EDProducer {
public:

  explicit TrackProducer(const edm::ParameterSet& iConfig);


  virtual void produce(edm::Event&, const edm::EventSetup&);
private:
  TrackProducerAlgorithm theAlgo;

};

#endif
