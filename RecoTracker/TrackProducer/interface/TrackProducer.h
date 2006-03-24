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
// $Id: TrackProducer.h,v 1.2 2006/03/22 15:16:18 tboccali Exp $
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

class Propagator;
class TrajectoryStateUpdator;
class MeasurementEstimator;


class TrackProducer : public edm::EDProducer {
public:
  explicit TrackProducer(const edm::ParameterSet&);

  ~TrackProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  //TrackCandidateCollection theTCCollection;//temporary: to be retrieved from the event

  TrackProducerAlgorithm theAlgo;
  edm::ParameterSet conf_;
  std::string src_;
};

#endif
