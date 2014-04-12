#ifndef CRackTrajectoryBuilder_h
#define CRackTrajectoryBuilder_h

//
// Package:         RecoTracker/SingleTrackPattern
// Class:           CRackTrajectoryBuilder
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


//to sort hits by the det position
 class CompareDetY_plus {
 public:
   CompareDetY_plus(const TrackerGeometry& tracker):_tracker(tracker){}
   bool operator()( const TrackingRecHit *rh1,
		    const TrackingRecHit *rh2)
   {
     const GeomDet* detPos1 = _tracker.idToDet(rh1->geographicalId());
     const GeomDet* detPos2 = _tracker.idToDet(rh2->geographicalId());

     GlobalPoint gp1 = detPos1->position();
     GlobalPoint gp2 = detPos2->position();
     
     if (gp1.y()>gp2.y())
       return true;
     if (gp1.y()<gp2.y())
       return false;
     //  if (gp1.y()== gp2.y())
     // 
     return (rh1->geographicalId() < rh2->geographicalId());
   };
    private:
   //   edm::ESHandle<TrackerGeometry> _tracker;
   const TrackerGeometry& _tracker;
 };

 class CompareDetY_minus {
 public:
   CompareDetY_minus(const TrackerGeometry& tracker):_tracker(tracker){}
   bool operator()( const TrackingRecHit *rh1,
		    const TrackingRecHit *rh2)
   {
     const GeomDet* detPos1 = _tracker.idToDet(rh1->geographicalId());
     const GeomDet* detPos2 = _tracker.idToDet(rh2->geographicalId());

     GlobalPoint gp1 = detPos1->position();
     GlobalPoint gp2 = detPos2->position();
     
     if (gp1.y()<gp2.y())
       return true;
     if (gp1.y()>gp2.y())
       return false;
     //  if (gp1.y()== gp2.y())
     // 
     return (rh1->geographicalId() < rh2->geographicalId());
   };
    private:
   //   edm::ESHandle<TrackerGeometry> _tracker;
   const TrackerGeometry& _tracker;
 };

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

class CRackTrajectoryBuilder 
{
//  using namespace std;

  typedef TrajectoryStateOnSurface     TSOS;
  typedef TrajectoryMeasurement        TM;
  
  typedef std::vector<const TrackingRecHit*>::iterator TrackingRecHitIterator;

  typedef std::pair<TrackingRecHitIterator, TrackingRecHitIterator> TrackingRecHitRange;
  typedef std::vector<TrackingRecHitRange>::iterator TrackingRecHitRangeIterator;

  //  typedef std::pair<TrackingRecHitIterator, TSOS> PairTrackingRecHitTsos; 
  typedef std::pair<TrackingRecHitRangeIterator, TSOS> PairTrackingRecHitTsos; 
  
 public:
  class CompareDetByTraj;
  friend class CompareDetByTraj;

  class CompareDetByTraj {
  public:
    CompareDetByTraj(const TSOS& tSos ):_tSos(tSos)
    {};
    bool operator()( const std::pair<TrackingRecHitRangeIterator, TSOS> rh1,
		     const std::pair<TrackingRecHitRangeIterator, TSOS> rh2)
    {
      GlobalPoint gp1 =  rh1.second.globalPosition();
      GlobalPoint gp2 =  rh2.second.globalPosition();

      GlobalPoint gpT = _tSos.globalPosition();
      GlobalVector gpDiff1 = gp1-gpT;
      GlobalVector gpDiff2 = gp2-gpT;

     //this might have a better performance ...
     //       float dist1 = ( gp1.x()-gpT.x() ) * ( gp1.x()-gpT.x() ) + ( gp1.y()-gpT.y() ) * ( gp1.y()-gpT.y() ) + ( gp1.z()-gpT.z() ) * ( gp1.z()-gpT.z() );
     //       float dist2 = ( gp2.x()-gpT.x() ) * ( gp2.x()-gpT.x() ) + ( gp2.y()-gpT.y() ) * ( gp2.y()-gpT.y() ) + ( gp2.z()-gpT.z() ) * ( gp2.z()-gpT.z() );
     //if ( dist1<dist2 )

     //     if ( gpDiff1.mag2() < gpDiff2.mag2() )
     

     float dist1 = gpDiff1 * _tSos.globalDirection();
     float dist2 = gpDiff2 * _tSos.globalDirection();

     if (dist1 < 0)
       return false;
     if ( dist1<dist2 )
       return true;
     
     return false;
   };
  private:
    const TrajectoryStateOnSurface& _tSos;
  };



 public:
  
  CRackTrajectoryBuilder(const edm::ParameterSet& conf);
  ~CRackTrajectoryBuilder();

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

    const TransientTrackingRecHitBuilder * hitBuilder() const {return RHBuilder;}

 private:
    std::vector<TrajectoryMeasurement> seedMeasurements(const TrajectorySeed& seed) const;
 
 
    std::vector<const TrackingRecHit*> SortHits(const SiStripRecHit2DCollection &collstereo,
					   const SiStripRecHit2DCollection &collrphi ,
					   const SiStripMatchedRecHit2DCollection &collmatched,
					   const SiPixelRecHitCollection &collpixel,
					   const TrajectorySeed &seed,
					   const bool bAddSeedHits
					   );

    
    //    std::vector<TrackingRecHitRange> SortByTrajectory (const std::vector<TrackingRecHitRange>& inputHits);


    TSOS startingTSOS(const TrajectorySeed& seed)const;
    void updateTrajectory( Trajectory& traj,
			   const TM& tm,
			   const TransientTrackingRecHit& hit) const;
    
    void AddHit(Trajectory &traj,
		const std::vector<const TrackingRecHit*>&Hits,
		Propagator *currPropagator
		);
    //		edm::OwnVector<TransientTrackingRecHit> hits);
    bool qualityFilter(const Trajectory& traj);

    bool isDifferentStripReHit2D  (const SiStripRecHit2D& hitA, const  SiStripRecHit2D& hitB );

    std::pair<TrajectoryStateOnSurface, const GeomDet*>
     innerState( const Trajectory& traj) const;


 
 private:
   edm::ESHandle<MagneticField> magfield;
   edm::ESHandle<TrackerGeometry> tracker;
   edm::ParameterSet conf_;
   
   PropagatorWithMaterial  *thePropagator;
   PropagatorWithMaterial  *thePropagatorOp;

//   AnalyticalPropagator *thePropagator;
//   AnalyticalPropagator *thePropagatorOp;

   KFUpdator *theUpdator;
   Chi2MeasurementEstimator *theEstimator;
   const TransientTrackingRecHitBuilder *RHBuilder;
   const KFTrajectorySmoother * theSmoother;
   const KFTrajectoryFitter * theFitter;
//   const KFTrajectoryFitter * theFitterOp;

   bool debug_info;
   bool fastPropagation;
   bool useMatchedHits;

   int theMinHits;
   double chi2cut;
   std::vector<Trajectory> trajFit;
   //RC edm::OwnVector<const TransientTrackingRecHit> hits;
   TransientTrackingRecHit::RecHitContainer  hits;
   bool seed_plus;
   std::string geometry;
//   TransientInitialStateEstimator*  theInitialState;
};

#endif
