#ifndef TrackProducerAlgorithm_h
#define TrackProducerAlgorithm_h

//
// Package:    RecoTracker/TrackProducer
// Class:      TrackProducerAlgorithm
// 
//
// Original Author:  Giuseppe Cerati
//         Created:  Thu Mar  9 17:29:31 CET 2006
// $Id: TrackProducerAlgorithm.h,v 1.3 2006/03/23 16:54:44 magni Exp $
//
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

class MagneticField;
class TrackingGeometry;
class TrajectoryFitter;
class Propagator;
class Trajectory;

typedef std::pair<Trajectory*, reco::Track*> AlgoProduct; 
typedef std::vector< AlgoProduct >  AlgoProductCollection;

class TrackProducerAlgorithm {
  
 public:

  TrackProducerAlgorithm(const edm::ParameterSet& conf) : 
    conf_(conf)
    { }

  ~TrackProducerAlgorithm() {}
  
  void run(const TrackingGeometry *, 
	   const MagneticField *, 
	   const TrackCandidateCollection&,
	   const TrajectoryFitter *,
	   const Propagator *,
	   AlgoProductCollection &);
  
 private:
  edm::ParameterSet conf_;
};

#endif
