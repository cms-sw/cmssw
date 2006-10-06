#ifndef GsfTrackProducerAlgorithm_h
#define GsfTrackProducerAlgorithm_h

//
// Package:    RecoTracker/TrackProducer
// Class:      GsfTrackProducerAlgorithm
// 
//
// Original Author:  Giuseppe Cerati
//         Created:  Thu Mar  9 17:29:31 CET 2006
// $Id: GsfTrackProducerAlgorithm.h,v 1.8 2006/07/28 15:17:29 cerati Exp $
//
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/GsfTrackExtra.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class MagneticField;
class TrackingGeometry;
class TrajectoryFitter;
class Propagator;
class Trajectory;
class TrajectoryStateOnSurface;
class TransientTrackingRecHitBuilder;


class GsfTrackProducerAlgorithm {
  
 public:
  typedef std::pair<Trajectory*, reco::GsfTrack*> AlgoProduct; 
  typedef std::vector< AlgoProduct >  AlgoProductCollection;

  GsfTrackProducerAlgorithm(const edm::ParameterSet& conf) : 
    conf_(conf)
    { }

  ~GsfTrackProducerAlgorithm() {}
  
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
};

#endif
