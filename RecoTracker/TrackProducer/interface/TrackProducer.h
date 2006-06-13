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
// $Id: TrackProducer.h,v 1.5 2006/05/30 14:36:20 cerati Exp $
//

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class TrackProducer : public TrackProducerBase, public edm::EDProducer {
public:

  explicit TrackProducer(const edm::ParameterSet& iConfig);


  virtual void produce(edm::Event&, const edm::EventSetup&);

  std::vector<reco::TransientTrack> getTransient(edm::Event&, const edm::EventSetup&);

private:
  TrackProducerAlgorithm theAlgo;

};

#endif
