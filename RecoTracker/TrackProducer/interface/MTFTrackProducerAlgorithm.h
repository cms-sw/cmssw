#ifndef MTFTrackProducerAlgorithm_h
#define MTFTrackProducerAlgorithm_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class MagneticField;
class TrackingGeometry;
class TrajectoryFitter;
class Trajectory;
class TrajectoryStateOnSurface;
class TransientTrackingRecHitBuilder;
class MultiRecHitCollector;
class SiTrackerMultiRecHitUpdatorMTF;
class TrajectoryMeasurement;
class MultiTrajectoryMeasurement;
class MultiTrackFilterHitCollector;

namespace reco{
	class Track;
}

class MTFTrackProducerAlgorithm {
   
   typedef std::pair<Trajectory*, std::pair<reco::Track*,PropagationDirection> > AlgoProduct;
   typedef std::vector< AlgoProduct > AlgoProductCollection;
   typedef MultiTrajectoryMeasurement MTM;
   typedef TrajectoryStateOnSurface TSOS;
 
 public:

  /// Constructor
  MTFTrackProducerAlgorithm(const edm::ParameterSet& pset):conf_(pset){}

  /// Destructor
  ~MTFTrackProducerAlgorithm() {}
  
  /// Run the Final Fit taking TrackCandidates as input
  void runWithCandidate(const TrackingGeometry *, 
			const MagneticField *, 
			const TrackCandidateCollection&,
			const TrajectoryFitter *,
			const TransientTrackingRecHitBuilder*,
			const MultiTrackFilterHitCollector* measurementTracker,
			const SiTrackerMultiRecHitUpdatorMTF*,
			const reco::BeamSpot&,
			AlgoProductCollection &) const;

 private:
  /// Construct Tracks to be put in the event
  bool buildTrack(const std::vector<Trajectory>&,
		  AlgoProductCollection& algoResults,
		  float,
		  const reco::BeamSpot&) const;

  /// accomplishes the fitting-smoothing step for each annealing value
  bool fit(const std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface>& hits,
           const TrajectoryFitter * theFitter,
           std::vector<Trajectory>& vtraj) const;

  //calculates the ndof according to the MTF prescription
  float calculateNdof(const std::vector<Trajectory>& vtraj) const;

  //creates MultiRecHits out of a KF trajectory 	
  std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface>
  collectHits(const std::map<int, std::vector<TrajectoryMeasurement> >& mapvtm,
              const MultiTrackFilterHitCollector* measurementCollector,
	      int i) const;

  //updates the hits with the specified annealing factor
  std::pair<TransientTrackingRecHit::RecHitContainer, TrajectoryStateOnSurface>
  updateHits(const std::map<int, std::vector<TrajectoryMeasurement> >& mapvtm,
	     const MultiTrackFilterHitCollector* measurementCollector,
             const SiTrackerMultiRecHitUpdatorMTF* updator,
             double annealing, 
	     const TransientTrackingRecHitBuilder* builder,
	     int i) const; 

  //removes from the trajectory isolated hits with very low weight
  void filter(const TrajectoryFitter* fitter, 
	      std::vector<Trajectory>& input, 
	      int minhits, std::vector<Trajectory>& output) const;

  edm::ParameterSet conf_;
};

#endif
