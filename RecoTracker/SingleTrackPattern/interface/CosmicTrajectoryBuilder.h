#ifndef CosmicTrajectoryBuilder_h
#define CosmicTrajectoryBuilder_h

//
// Package:         RecoTracker/SingleTrackPattern
// Class:           CosmicTrajectoryBuilder
// Original Author:  Michele Pioppi-INFN perugia

#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

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
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#ifndef TrajectoryBuilder_CompareHitY
#define TrajectoryBuilder_CompareHitY

class CompareHitY {
public:
  CompareHitY(const TrackerGeometry &tracker) : _tracker(tracker) {}
  bool operator()(const TrackingRecHit *rh1, const TrackingRecHit *rh2) {
    GlobalPoint gp1 = _tracker.idToDet(rh1->geographicalId())->surface().toGlobal(rh1->localPosition());
    GlobalPoint gp2 = _tracker.idToDet(rh2->geographicalId())->surface().toGlobal(rh2->localPosition());
    return gp1.y() < gp2.y();
  };

private:
  //   edm::ESHandle<TrackerGeometry> _tracker;
  const TrackerGeometry &_tracker;
};

class CompareHitY_plus {
public:
  CompareHitY_plus(const TrackerGeometry &tracker) : _tracker(tracker) {}
  bool operator()(const TrackingRecHit *rh1, const TrackingRecHit *rh2) {
    GlobalPoint gp1 = _tracker.idToDet(rh1->geographicalId())->surface().toGlobal(rh1->localPosition());
    GlobalPoint gp2 = _tracker.idToDet(rh2->geographicalId())->surface().toGlobal(rh2->localPosition());
    return gp1.y() > gp2.y();
  };

private:
  //   edm::ESHandle<TrackerGeometry> _tracker;
  const TrackerGeometry &_tracker;
};

#endif

class CosmicTrajectoryBuilder {
  typedef TrajectoryStateOnSurface TSOS;
  typedef TrajectoryMeasurement TM;

public:
  CosmicTrajectoryBuilder(const edm::ParameterSet &conf, edm::ConsumesCollector iC);
  ~CosmicTrajectoryBuilder();

  /// @brief Runs the algorithm. Input collections are allowed to
  /// be missing, in that case they will be skipped.
  /// @param collseed: input seeds to process
  /// @param collstereo: Stereo strip hit collection
  /// @param collrphi: rphi hit collection
  /// @param collmatched: matched hit collection
  /// @param collpixel: Pixel hit collection
  /// @param p2RecHits1D: Phase-2 OT hit collection
  /// @param es: EventSetup to use
  /// @param e: Event to use
  /// @param [out] trajoutput: Will be populated with trajectory candidates.
  void run(const TrajectorySeedCollection &collseed,
           std::optional<std::reference_wrapper<const SiStripRecHit2DCollection> > collstereo,
           std::optional<std::reference_wrapper<const SiStripRecHit2DCollection> > collrphi,
           std::optional<std::reference_wrapper<const SiStripMatchedRecHit2DCollection> > collmatched,
           std::optional<std::reference_wrapper<const SiPixelRecHitCollection> > collpixel,
           std::optional<std::reference_wrapper<const Phase2TrackerRecHit1DCollectionNew> > p2RecHits1D,
           const edm::EventSetup &es,
           edm::Event &e,
           std::vector<Trajectory> &trajoutput);

  void init(const edm::EventSetup &es, bool);
  Trajectory createStartingTrajectory(const TrajectorySeed &seed) const;

  const TransientTrackingRecHitBuilder *hitBuilder() const { return RHBuilder; }

private:
  std::vector<TrajectoryMeasurement> seedMeasurements(const TrajectorySeed &seed) const;

  /// @brief Collects all hits and sorts them in global y,
  /// either descending or ascending depending on the seed.
  /// Inputs are allowed to be missing, in this case
  /// they will be skipped.
  /// @param collseed: input seeds to process
  /// @param collstereo: Stereo strip hit collection
  /// @param collrphi: rphi hit collection
  /// @param collmatched: matched hit collection
  /// @param collpixel: Pixel hit collection
  /// @param p2RecHits1D: Phase-2 OT hit collection
  /// @param seed: The seed to work with - hits will be removed
  /// @return a y-sorted vector with all hits in the input collections, except those already on the seed.
  std::vector<const TrackingRecHit *> SortHits(
      std::optional<std::reference_wrapper<const SiStripRecHit2DCollection> > collstereo,
      std::optional<std::reference_wrapper<const SiStripRecHit2DCollection> > collrphi,
      std::optional<std::reference_wrapper<const SiStripMatchedRecHit2DCollection> > collmatched,
      std::optional<std::reference_wrapper<const SiPixelRecHitCollection> > collpixel,
      std::optional<std::reference_wrapper<const Phase2TrackerRecHit1DCollectionNew> > p2RecHits1D,
      const TrajectorySeed &seed);

  TSOS startingTSOS(const TrajectorySeed &seed) const;
  void updateTrajectory(Trajectory &traj, const TM &tm, const TransientTrackingRecHit &hit) const;

  void AddHit(Trajectory &traj, const std::vector<const TrackingRecHit *> &Hits);
  //		edm::OwnVector<TransientTrackingRecHit> hits);
  bool qualityFilter(const Trajectory &traj);

private:
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> builderToken_;
  const MagneticField *magfield;
  const TrackerGeometry *tracker;

  PropagatorWithMaterial *thePropagator;
  PropagatorWithMaterial *thePropagatorOp;
  KFUpdator *theUpdator;
  Chi2MeasurementEstimator *theEstimator;
  const TransientTrackingRecHitBuilder *RHBuilder;
  TkClonerImpl hitCloner;
  KFTrajectorySmoother *theSmoother;
  KFTrajectoryFitter *theFitter;

  int theMinHits;
  double chi2cut;
  std::vector<Trajectory> trajFit;
  //RC edm::OwnVector<const TransientTrackingRecHit> hits;
  TransientTrackingRecHit::RecHitContainer hits;
  bool seed_plus;
  std::string geometry;
};

#endif
