#ifndef TrackProducerAlgorithm_h
#define TrackProducerAlgorithm_h

/** \class TrackProducerAlgorithm
 *  This class calls the Final Fit and builds the Tracks then produced by the TrackProducer or by the TrackRefitter
 *
 *  $Date: 2008/11/21 15:09:06 $
 *  $Revision: 1.21 $
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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

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
  typedef edm::RefToBase<TrajectorySeed> SeedRef;
  typedef edm::AssociationMap<edm::OneToOne<std::vector<T>,std::vector<VertexConstraint> > > 
  VtxConstraintAssociationCollection;
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
			const reco::BeamSpot&,
			AlgoProductCollection &);

  /// Run the Final Fit taking Tracks as input (for Refitter)
  void runWithTrack(const TrackingGeometry *, 
		    const MagneticField *, 
		    const TrackCollection&,
		    const TrajectoryFitter *,
		    const Propagator *,
		    const TransientTrackingRecHitBuilder*,
		    const reco::BeamSpot&,
		    AlgoProductCollection &);

  /// Run the Final Fit taking TrackMomConstraintAssociation as input (Refitter with momentum constraint)
  void runWithMomentum(const TrackingGeometry *, 
		       const MagneticField *, 
		       const TrackMomConstraintAssociationCollection&,
		       const TrajectoryFitter *,
		       const Propagator *,
		       const TransientTrackingRecHitBuilder*,
		       const reco::BeamSpot&,
		       AlgoProductCollection &);

  /// Run the Final Fit taking TrackVtxConstraintAssociation as input (Refitter with vertex constraint)
  ///   currently hit sorting is disabled - will work (only) with standard tracks
  void runWithVertex(const TrackingGeometry *, 
		     const MagneticField *, 
		     const VtxConstraintAssociationCollection&,
		     const TrajectoryFitter *,
		     const Propagator *,
		     const TransientTrackingRecHitBuilder*,
		     const reco::BeamSpot&,
		     AlgoProductCollection &);

  /// Construct Tracks to be put in the event
  bool buildTrack(const TrajectoryFitter *,
		  const Propagator *,
		  AlgoProductCollection& ,
		  TransientTrackingRecHit::RecHitContainer&,
		  TrajectoryStateOnSurface& ,
		  const TrajectorySeed&,		  
		  float,
		  const reco::BeamSpot&,
		  SeedRef seedRef = SeedRef(),
		  int qualityMask=0);

 private:
  edm::ParameterSet conf_;  
  std::string algoName_;

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
						float,
						const reco::BeamSpot&,
						SeedRef seedRef,
						int qualityMask);


template <> bool
TrackProducerAlgorithm<reco::GsfTrack>::buildTrack(const TrajectoryFitter *,
						   const Propagator *,
						   AlgoProductCollection& ,
						   TransientTrackingRecHit::RecHitContainer&,
						   TrajectoryStateOnSurface& ,
						   const TrajectorySeed&,
						   float,
						   const reco::BeamSpot&,
						   SeedRef seedRef,
						   int qualityMask);

#endif
