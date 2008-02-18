#ifndef DAFTrackProducerAlgorithm_h
#define DAFTrackProducerAlgorithm_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

class MagneticField;
class TrackingGeometry;
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
  
 public:

  /// Constructor
  DAFTrackProducerAlgorithm(const edm::ParameterSet& pset):conf_(pset){}

  /// Destructor
  ~DAFTrackProducerAlgorithm() {}
  
  /// Run the Final Fit taking TrackCandidates as input
  void runWithCandidate(const TrackingGeometry *, 
			const MagneticField *, 
			const TrackCandidateCollection&,
			const TrajectoryFitter *,
			const TransientTrackingRecHitBuilder*,
			const MultiRecHitCollector* measurementTracker,
			const SiTrackerMultiRecHitUpdator*,
			AlgoProductCollection &) const;

 private:
  /// Construct Tracks to be put in the event
  bool buildTrack(const std::vector<Trajectory>&,
		  AlgoProductCollection& algoResults,
		  float) const;

  /// accomplishes the fitting-smoothing step for each annealing value
  void fit(const std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface>& hits,
           const TrajectoryFitter * theFitter,
           std::vector<Trajectory>& vtraj) const;

  //calculates the ndof according to the DAF prescription
  float calculateNdof(const std::vector<Trajectory>& vtraj) const;

  //creates MultiRecHits out of a KF trajectory 	
  std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface>
  collectHits(const std::vector<Trajectory>& vtraj,
              const MultiRecHitCollector* measurementCollector) const;

  //updates the hits with the specified annealing factor
  std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface>
  updateHits(const std::vector<Trajectory>& vtraj,
             const SiTrackerMultiRecHitUpdator* updator,
             double annealing) const; 

  //removes from the trajectory isolated hits with very low weight
  void filter(const TrajectoryFitter* fitter, 
	      std::vector<Trajectory>& input, 
	      int minhits, std::vector<Trajectory>& output) const;

  edm::ParameterSet conf_;
};

#endif
