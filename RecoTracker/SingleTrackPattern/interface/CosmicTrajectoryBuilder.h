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
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DMatchedLocalPosCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
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
class CosmicTrajectoryBuilder 
{

  typedef TrajectoryStateOnSurface     TSOS;
  typedef TrajectoryMeasurement        TM;

 public:
  
  CosmicTrajectoryBuilder(const edm::ParameterSet& conf);
  ~CosmicTrajectoryBuilder();

  /// Runs the algorithm
    void run(const TrajectorySeedCollection &collseed,
	     const SiStripRecHit2DLocalPosCollection &collstereo,
	     const SiStripRecHit2DLocalPosCollection &collrphi ,
	     const SiStripRecHit2DMatchedLocalPosCollection &collmatched,
	     const SiPixelRecHitCollection &collpixel,
	     const edm::EventSetup& es,
	     TrackCandidateCollection &output);
    void init(const edm::EventSetup& es);
 private:
    std::vector<TrajectoryMeasurement> seedMeasurements(const TrajectorySeed& seed) const;
    Trajectory createStartingTrajectory( const TrajectorySeed& seed) const;
 
    vector<const TrackingRecHit*> SortHits(const SiStripRecHit2DLocalPosCollection &collstereo,
					   const SiStripRecHit2DLocalPosCollection &collrphi ,
					   const SiStripRecHit2DMatchedLocalPosCollection &collmatched,
					   const SiPixelRecHitCollection &collpixel);

    TSOS startingTSOS(const TrajectorySeed& seed)const;
    void updateTrajectory( Trajectory& traj,
			   const TM& tm) const;
    
    void AddHit(Trajectory traj,
		edm::OwnVector<TransientTrackingRecHit> hits);
    bool qualityFilter(Trajectory traj);
 private:
   edm::ESHandle<MagneticField> magfield;
   edm::ESHandle<GeometricSearchTracker> track;
   edm::ESHandle<TrackerGeometry> tracker;
   edm::ParameterSet conf_;
   TrajectoryStateTransform tsTransform;
   AnalyticalPropagator  *thePropagator;
   KFUpdator *theUpdator;
   Chi2MeasurementEstimator *theEstimator;
   TkTransientTrackingRecHitBuilder *RHBuilder;
   const KFTrajectoryFitter * theFitter;
   const LayerMeasurements*      theLayerMeasurements;
   Trajectory *cachetraj;
   vector<BarrelDetLayer*> bl;
   int theMinHits;
   bool chi2cut;


};

#endif
