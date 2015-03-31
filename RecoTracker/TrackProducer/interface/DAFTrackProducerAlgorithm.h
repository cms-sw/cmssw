/** \class DAFTrackProducerAlgorithm
  *  All is needed to run the deterministic annealing algorithm. Ported from ORCA. 
  *
  *  \author tropiano, genta
  *  \review in May 2014 by brondolin 
  */

#ifndef DAFTrackProducerAlgorithm_h
#define DAFTrackProducerAlgorithm_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class MagneticField;
class TrackingGeometry;
class TrajAnnealing;
class TrajectoryFitter;
class Trajectory;
class TrajectoryStateOnSurface;
class TransientTrackingRecHitBuilder;
class MultiRecHitCollector;
class SiTrackerMultiRecHitUpdator;
namespace reco{
	class Track;
}

class DAFTrackProducerAlgorithm {

   typedef std::pair<Trajectory*, std::pair<reco::Track*,PropagationDirection> > AlgoProduct;
   typedef std::vector< AlgoProduct >  AlgoProductCollection;
   typedef std::vector<TrajAnnealing> TrajAnnealingCollection;
  
 public:

  DAFTrackProducerAlgorithm(const edm::ParameterSet& conf);
  ~DAFTrackProducerAlgorithm() {}
  
  /// Run the Final Fit taking TrackCandidates as input
  void runWithCandidate(const TrackingGeometry *, 
			const MagneticField *, 
			//const TrackCandidateCollection&,
			const TrajTrackAssociationCollection&,
			const MeasurementTrackerEvent *measTk,
			const TrajectoryFitter *,
			const TransientTrackingRecHitBuilder*,
			const MultiRecHitCollector* measurementTracker,
			const SiTrackerMultiRecHitUpdator*,
			const reco::BeamSpot&,
			AlgoProductCollection &,
			TrajAnnealingCollection &,
			bool ,
			AlgoProductCollection& , 
			AlgoProductCollection&) const;

 private:
  /// Construct Tracks to be put in the event
  bool buildTrack(const Trajectory,
		  AlgoProductCollection& algoResults,
		  float,
		  const reco::BeamSpot&, 
		  const reco::TrackRef* ) const;

  /// accomplishes the fitting-smoothing step for each annealing value
  Trajectory fit(const std::pair<TransientTrackingRecHit::RecHitContainer,
                                    TrajectoryStateOnSurface>& hits,
                                    const TrajectoryFitter * theFitter,
                                    Trajectory vtraj) const;

  //calculates the ndof according to the DAF prescription
  float calculateNdof(const Trajectory vtraj) const;

  //creates MultiRecHits out of a KF trajectory 	
  std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface> collectHits(
              const Trajectory vtraj,
              const MultiRecHitCollector* measurementCollector,
              const MeasurementTrackerEvent *measTk     ) const;

  //updates the hits with the specified annealing factor
  std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface> updateHits(
	      const Trajectory vtraj,
              const SiTrackerMultiRecHitUpdator* updator,
	      const MeasurementTrackerEvent* theMTE,
              double annealing) const; 

  //removes from the trajectory isolated hits with very low weight
  void filter(const TrajectoryFitter* fitter, 
	      std::vector<Trajectory>& input, 
	      int minhits, std::vector<Trajectory>& output,
	      const TransientTrackingRecHitBuilder* builder) const;

  int countingGoodHits(const Trajectory traj) const;

  int checkHits( Trajectory iInitTraj, const Trajectory iFinalTraj) const; 

  void PrintHit(const TrackingRecHit* const& hit, TrajectoryStateOnSurface& tsos) const;

  edm::ParameterSet conf_;
  int minHits_;  
};

#endif
