#ifndef TrackCandidateMakerAlgorithm_h
#define TrackCandidateMakerAlgorithm_h

//
// Package:         RecoTracker/RoadSearchTrackCandidateMaker
// Class:           RoadSearchTrackCandidateMakerAlgorithm
// 
// Description:     Converts cleaned clouds into
//                  TrackCandidates using the 
//                  TrajectoryBuilder framework
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Wed Mar 15 13:00:00 UTC 2006
//
// $Author: gutsche $
// $Date: 2007/07/10 03:05:52 $
// $Revision: 1.12 $
//

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

class TrajectoryStateUpdator;
class MeasurementEstimator;
class PropagatorWithMaterial;
class SteppingHelixPropagator;


class RoadSearchTrackCandidateMakerAlgorithm 
{
 public:
  
  RoadSearchTrackCandidateMakerAlgorithm(const edm::ParameterSet& conf);
  ~RoadSearchTrackCandidateMakerAlgorithm();

  /// Runs the algorithm
  void run(const RoadSearchCloudCollection* input,
	   const edm::Event& e,
	   const edm::EventSetup& es,
	   TrackCandidateCollection &output);

  //  edm::OwnVector<TrackingRecHit> 
  std::vector<TrajectoryMeasurement>  FindBestHitsByDet(const TrajectoryStateOnSurface& tsosBefore,
		      const std::set<const GeomDet*>& theDets,
		      edm::OwnVector<TrackingRecHit>& theHits);


  std::vector<TrajectoryMeasurement>  FindBestHit(const TrajectoryStateOnSurface& tsosBefore,
				     const std::set<const GeomDet*>& theDets,
				     edm::OwnVector<TrackingRecHit>& theHits);


  std::vector<TrajectoryMeasurement>  FindBestHits(const TrajectoryStateOnSurface& tsosBefore,
                                     const std::set<const GeomDet*>& theDets,
				     const SiStripRecHitMatcher* theHitMatcher,
                                     edm::OwnVector<TrackingRecHit>& theHits);

  bool chooseStartingLayers( RoadSearchCloud::RecHitVector& recHits, int layer0,
			     const std::multimap<int, const DetLayer*>& layer_map,
			     std::set<const DetLayer*>& good_layers,
			     std::vector<const DetLayer*>& middle_layers ,
			     RoadSearchCloud::RecHitVector& recHits_middle);


  FreeTrajectoryState initialTrajectory(const edm::EventSetup& es,
					const TrackingRecHit* InnerHit, 
					const TrackingRecHit* OuterHit);

  std::vector<Trajectory> extrapolateTrajectories(std::vector<Trajectory>& inputTrajectories,
						  RoadSearchCloud::RecHitVector& theLayerHits,
						  const DetLayer* innerHitLayer,
						  const DetLayer* outerHitLayer);

 private:
  edm::ParameterSet conf_;

  unsigned int theNumHitCut;
  double theChi2Cut;
  bool NoFieldCosmic_;
  int MinChunkLength_;
  int nFoundMin_;




  std::string measurementTrackerName_;
  
  bool debug_;
  
  const MeasurementTracker*  theMeasurementTracker;
  const TrackerGeometry* trackerGeom;
  const TransientTrackingRecHitBuilder* ttrhBuilder;
  const MagneticField* magField;
  
  PropagatorWithMaterial* thePropagator;
  PropagatorWithMaterial* theRevPropagator;
  TrajectoryStateUpdator* theUpdator;
  MeasurementEstimator* theEstimator;
  SiStripRecHitMatcher* theHitMatcher;

    const DetLayer* layers[128];
    bool lstereo[128];
    int nhits_l[128];
    int nlayers ;

};

#endif
