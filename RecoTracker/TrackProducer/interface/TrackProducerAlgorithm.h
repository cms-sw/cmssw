#ifndef TrackProducerAlgorithm_h
#define TrackProducerAlgorithm_h

//
// Package:    RecoTracker/TrackProducer
// Class:      TrackProducerAlgorithm
// 
//
// Original Author:  Giuseppe Cerati
//         Created:  Thu Mar  9 17:29:31 CET 2006
// $Id: TrackProducerAlgorithm.h,v 1.8 2006/07/28 15:17:29 cerati Exp $
//
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class MagneticField;
class TrackingGeometry;
class TrajectoryFitter;
class Propagator;
class Trajectory;
class TrajectoryStateOnSurface;
class TransientTrackingRecHitBuilder;

typedef std::pair<Trajectory*, reco::Track*> AlgoProduct; 
typedef std::vector< AlgoProduct >  AlgoProductCollection;

class TrackProducerAlgorithm {
  
 public:

  TrackProducerAlgorithm(const edm::ParameterSet& conf) : 
    conf_(conf),
    numberOfTrackParameters_(conf_.getParameter<int>("numberOfTrackParameters"))
    { }

  ~TrackProducerAlgorithm() {}
  
  void runWithCandidate(const TrackingGeometry *, 
			const MagneticField *, 
			const TrackCandidateCollection&,
			const TrajectoryFitter *,
			const Propagator *,
			const TransientTrackingRecHitBuilder*,
			AlgoProductCollection &);

  void runWithTrack(const TrackingGeometry *, 
		    const MagneticField *, 
		    const reco::TrackCollection&,
		    const TrajectoryFitter *,
		    const Propagator *,
		    const TransientTrackingRecHitBuilder*,
		    AlgoProductCollection &);

  bool buildTrack(const TrajectoryFitter *,
		  const Propagator *,
		  AlgoProductCollection& ,
		  TransientTrackingRecHit::RecHitContainer&,
		  TrajectoryStateOnSurface& ,
		  const TrajectorySeed&,
		  float);

 private:
  edm::ParameterSet conf_;
  int numberOfTrackParameters_;
};

#endif
