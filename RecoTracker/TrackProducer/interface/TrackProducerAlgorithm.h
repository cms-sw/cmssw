#ifndef TrackProducerAlgorithm_h
#define TrackProducerAlgorithm_h

/** \class TrackProducerAlgorithm
 *  This class calls the Final Fit and builds the Tracks then produced by the TrackProducer or by the TrackRefitter
 *
 *  $Date: 2007/07/30 23:32:28 $
 *  $Revision: 1.13 $
 *  \author cerati
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/TrackConstraintAssociation.h"

class MagneticField;
class TrackingGeometry;
class TrajectoryFitter;
class Propagator;
class Trajectory;
class TrajectoryStateOnSurface;
class TransientTrackingRecHitBuilder;


template <class T>
class TrackProducerAlgorithm {
public:
  typedef std::vector<T> TrackCollection;
  typedef std::pair<Trajectory*, std::pair<T*,PropagationDirection> > AlgoProduct; 
  typedef std::vector< AlgoProduct >  AlgoProductCollection;


 public:

  /// Constructor
  TrackProducerAlgorithm(const edm::ParameterSet& conf) : 
    conf_(conf)
    { }

  /// Destructor
  ~TrackProducerAlgorithm() {}
  
  /// Run the Final Fit taking TrackCandidates as input
  void runWithCandidate(const TrackingGeometry *, 
			const MagneticField *, 
			const TrackCandidateCollection&,
			const TrajectoryFitter *,
			const Propagator *,
			const TransientTrackingRecHitBuilder*,
			AlgoProductCollection &);

  /// Run the Final Fit taking Tracks as input (for Refitter)
  void runWithTrack(const TrackingGeometry *, 
		    const MagneticField *, 
		    const TrackCollection&,
		    const TrajectoryFitter *,
		    const Propagator *,
		    const TransientTrackingRecHitBuilder*,
		    AlgoProductCollection &);

  /// Run the Final Fit taking TrackMomConstraintAssociation as input (Refitter with momentum constraint)
  void runWithMomentum(const TrackingGeometry *, 
		       const MagneticField *, 
		       const TrackMomConstraintAssociationCollection&,
		       const TrajectoryFitter *,
		       const Propagator *,
		       const TransientTrackingRecHitBuilder*,
		       AlgoProductCollection &);

  /// Run the Final Fit taking TrackVtxConstraintAssociation as input (Refitter with vertex constraint)
  void runWithVertex(const TrackingGeometry *, 
		     const MagneticField *, 
		     const TrackVtxConstraintAssociationCollection&,
		     const TrajectoryFitter *,
		     const Propagator *,
		     const TransientTrackingRecHitBuilder*,
		     AlgoProductCollection &);

  /// Construct Tracks to be put in the event
  bool buildTrack(const TrajectoryFitter *,
		  const Propagator *,
		  AlgoProductCollection& ,
		  TransientTrackingRecHit::RecHitContainer&,
		  TrajectoryStateOnSurface& ,
		  const TrajectorySeed&,
		  float);

 private:
  edm::ParameterSet conf_;
  TransientTrackingRecHit::RecHitContainer getHitVector(const T *, PropagationDirection&, float&,
							const TransientTrackingRecHitBuilder*);
  TrajectoryStateOnSurface getInitialState(const T * theT,
					   TransientTrackingRecHit::RecHitContainer& hits,
					   const TrackingGeometry * theG,
					   const MagneticField * theMF);

};

#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.icc"

template <> bool
TrackProducerAlgorithm<reco::Track>::buildTrack(const TrajectoryFitter *,
						   const Propagator *,
						   AlgoProductCollection& ,
						   TransientTrackingRecHit::RecHitContainer&,
						   TrajectoryStateOnSurface& ,
						   const TrajectorySeed&,
						   float);

template <> bool
TrackProducerAlgorithm<reco::GsfTrack>::buildTrack(const TrajectoryFitter *,
						      const Propagator *,
						      AlgoProductCollection& ,
						      TransientTrackingRecHit::RecHitContainer&,
						      TrajectoryStateOnSurface& ,
						      const TrajectorySeed&,
						      float);

#endif
