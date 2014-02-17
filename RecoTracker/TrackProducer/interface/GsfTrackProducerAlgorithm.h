#ifndef GsfTrackProducerAlgorithm_h
#define GsfTrackProducerAlgorithm_h

//
// Package:    RecoTracker/TrackProducer
// Class:      GsfTrackProducerAlgorithm
// 
//
// Original Author:  Giuseppe Cerati
//         Created:  Thu Mar  9 17:29:31 CET 2006
// $Id: GsfTrackProducerAlgorithm.h,v 1.4 2007/08/23 20:59:33 ratnik Exp $
//
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
// #include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
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
  typedef std::pair<Trajectory*, std::pair<reco::GsfTrack*,PropagationDirection> > AlgoProduct; 
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
