#ifndef CosmicTrajectoryBuilder_h
#define CosmicTrajectoryBuilder_h

//
// Package:         RecoTracker/SingleTrackPattern
// Class:           CosmicTrajectoryBuilder
// Original Author:  Michele Pioppi-INFN perugia

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "FWCore/Framework/interface/ESHandle.h"

#ifndef TrajectoryBuilder_CompareHitY
#define TrajectoryBuilder_CompareHitY

class CompareHitY {
 public:
   CompareHitY(const TrackerGeometry& tracker):_tracker(tracker){}
   bool operator()( const TrackingRecHit *rh1,
		    const TrackingRecHit *rh2)
   {
     GlobalPoint gp1=_tracker.idToDet(rh1->geographicalId())->surface().toGlobal(rh1->localPosition());
     GlobalPoint gp2=_tracker.idToDet(rh2->geographicalId())->surface().toGlobal(rh2->localPosition());
     return gp1.y()<gp2.y();};
 private:
   //   edm::ESHandle<TrackerGeometry> _tracker;
   const TrackerGeometry& _tracker;
};
 
class CompareHitY_plus {
 public:
   CompareHitY_plus(const TrackerGeometry& tracker):_tracker(tracker){}
   bool operator()( const TrackingRecHit *rh1,
		    const TrackingRecHit *rh2)
   {
     GlobalPoint gp1=_tracker.idToDet(rh1->geographicalId())->surface().toGlobal(rh1->localPosition());
     GlobalPoint gp2=_tracker.idToDet(rh2->geographicalId())->surface().toGlobal(rh2->localPosition());
     return gp1.y()>gp2.y();};
 private:
   //   edm::ESHandle<TrackerGeometry> _tracker;
   const TrackerGeometry& _tracker;
};
 
#endif

class CosmicTrajectoryBuilder 
{

  typedef TrajectoryStateOnSurface     TSOS;
  typedef TrajectoryMeasurement        TM;

 public:
  
  CosmicTrajectoryBuilder(const edm::ParameterSet& conf);
  ~CosmicTrajectoryBuilder();

  /// Runs the algorithm
    
    void run(const TrajectorySeedCollection &collseed,
	     const SiStripRecHit2DCollection &collstereo,
	     const SiStripRecHit2DCollection &collrphi ,
	     const SiStripMatchedRecHit2DCollection &collmatched,
	     const SiPixelRecHitCollection &collpixel,
	     const edm::EventSetup& es,
	     edm::Event& e,
	     std::vector<Trajectory> &trajoutput);

    void init(const edm::EventSetup& es,bool);
    Trajectory createStartingTrajectory( const TrajectorySeed& seed) const;
 private:
    std::vector<TrajectoryMeasurement> seedMeasurements(const TrajectorySeed& seed) const;
 
 
    std::vector<const TrackingRecHit*> SortHits(const SiStripRecHit2DCollection &collstereo,
					   const SiStripRecHit2DCollection &collrphi ,
					   const SiStripMatchedRecHit2DCollection &collmatched,
					   const SiPixelRecHitCollection &collpixel,
					   const TrajectorySeed &seed);

    TSOS startingTSOS(const TrajectorySeed& seed)const;
    void updateTrajectory( Trajectory& traj,
			   const TM& tm,
			   const TransientTrackingRecHit& hit) const;
    
    void AddHit(Trajectory &traj,
		const std::vector<const TrackingRecHit*>&Hits);
    //		edm::OwnVector<TransientTrackingRecHit> hits);
    bool qualityFilter(const Trajectory& traj);


 
 private:
   edm::ESHandle<MagneticField> magfield;
   edm::ESHandle<TrackerGeometry> tracker;
   edm::ParameterSet conf_;
   
   PropagatorWithMaterial  *thePropagator;
   PropagatorWithMaterial  *thePropagatorOp;
   KFUpdator *theUpdator;
   Chi2MeasurementEstimator *theEstimator;
   const TransientTrackingRecHitBuilder *RHBuilder;
   const KFTrajectorySmoother * theSmoother;
   const KFTrajectoryFitter * theFitter;
 
 

   int theMinHits;
   double chi2cut;
   std::vector<Trajectory> trajFit;
   //RC edm::OwnVector<const TransientTrackingRecHit> hits;
   TransientTrackingRecHit::RecHitContainer  hits;
   bool seed_plus;
   std::string geometry;
};

#endif
