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
// $Id: TrackProducer.h,v 1.3 2006/03/24 10:35:33 magni Exp $
//

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

class TrackProducer : public TrackProducerBase, public edm::EDProducer {
public:

  explicit TrackProducer(const edm::ParameterSet& iConfig);

/*   ~TrackProducer(); */
  
/*   virtual void getFromES(const edm::EventSetup&, */
/* 			 edm::ESHandle<TrackerGeometry>& , */
/* 			 edm::ESHandle<MagneticField>& , */
/* 			 edm::ESHandle<TrajectoryFitter>& , */
/* 			 edm::ESHandle<Propagator>& ); */

/*   virtual void getFromEvt(edm::Event&, edm::Handle<TrackCandidateCollection>&); */

/*   virtual void putInEvt(edm::Event&, */
/* 			std::auto_ptr<TrackingRecHitCollection>&, */
/* 			std::auto_ptr<reco::TrackCollection>&, */
/* 			std::auto_ptr<reco::TrackExtraCollection>&, */
/* 			AlgoProductCollection&); */

  virtual void produce(edm::Event&, const edm::EventSetup&);

private:
  //TrackCandidateCollection theTCCollection;//temporary: to be retrieved from the event

  TrackProducerAlgorithm theAlgo;
/*   edm::ParameterSet conf_; */
/*   std::string src_; */
};

#endif
