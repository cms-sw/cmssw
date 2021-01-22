#ifndef TrackingTools_TrackRefitter_TrackTransformer_H
#define TrackingTools_TrackRefitter_TrackTransformer_H

/** \class TrackTransformer
 *  This class takes a reco::Track and refits the rechits inside it.
 *  The final result is a Trajectory refitted and smoothed.
 *  To make the refitting (and the smoothing) the usual KF tools are used.
 *
 *  CAVEAT: till now (it will be changed in the near future) the class stores the
 *  pointers to the services, therefore EACH event the setServices(const edm::EventSetup&)
 *  method MUST be called in the code in which the TrackTransformer is used.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "TrackingTools/TrackRefitter/interface/TrackTransformerBase.h"

#include "TrackingTools/TrackRefitter/interface/RefitDirection.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
  class TransientTrack;
}

class TrajectoryFitter;
class TrajectorySmoother;
class Propagator;
class TransientTrackingRecHitBuilder;
class Trajectory;
class GlobalTrackingGeometryRecord;
class IdealMagneticFieldRecord;
class TrajectoryFitterRecord;
class TrackingComponentsRecord;
class TransientRecHitRecord;

class TrackTransformer final : public TrackTransformerBase {
public:
  /// Constructor (for modules not yet migrated to ES-consumes)
  explicit TrackTransformer(const edm::ParameterSet&);

  /// Constructor (for modules migrated to ES-consumes)
  explicit TrackTransformer(const edm::ParameterSet&, edm::ConsumesCollector&);
  explicit TrackTransformer(const edm::ParameterSet& parameterSet, edm::ConsumesCollector&& iC)
      : TrackTransformer(parameterSet, iC) {}

  /// Destructor
  ~TrackTransformer() override;

  /// fillDescriptions
  static void fillPSetDescription(edm::ParameterSetDescription& descriptions,
                                  bool DoPredictionsOnly = false,
                                  const std::string& Fitter = "KFFitterForRefitInsideOut",
                                  const std::string& Smoother = "KFSmootherForRefitInsideOut",
                                  const std::string& Propagator = "SmartPropagatorAnyRK",
                                  const std::string& RefitDirection = "alongMomentum",
                                  bool RefitRPCHits = true,
                                  const std::string& TrackerRecHitBuilder = "WithTrackAngle",
                                  const std::string& MuonRecHitBuilder = "MuonRecHitBuilder",
                                  const std::string& MTDRecHitBuilder = "MTDRecHitBuilder");

  // Operations

  /// Convert a reco::Track into Trajectory
  std::vector<Trajectory> transform(const reco::Track&) const override;

  /// Convert a reco::TrackRef into Trajectory
  std::vector<Trajectory> transform(const reco::TrackRef&) const;

  /// Convert a reco::TrackRef into Trajectory, refit with a new set of hits
  std::vector<Trajectory> transform(const reco::TransientTrack&, TransientTrackingRecHit::ConstRecHitContainer&) const;

  /// the magnetic field
  const MagneticField* magneticField() const { return &*theMGField; }

  /// the tracking geometry
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry() const { return theTrackingGeometry; }

  /// set the services needed by the TrackTransformer
  void setServices(const edm::EventSetup&) override;

  /// the refitter used to refit the reco::Track
  std::unique_ptr<TrajectoryFitter> const& refitter() const { return theFitter; }

  /// the smoother used to smooth the trajectory which came from the refitting step
  std::unique_ptr<TrajectorySmoother> const& smoother() const { return theSmoother; }

  TransientTrackingRecHit::ConstRecHitContainer getTransientRecHits(const reco::TransientTrack& track) const;

private:
  RefitDirection::GeometricalDirection checkRecHitsOrdering(TransientTrackingRecHit::ConstRecHitContainer const&) const;

  unsigned long long theCacheId_TRH = 0;

  const bool theRPCInTheFit;

  const bool theDoPredictionsOnly;
  const RefitDirection theRefitDirection;

  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> theTrackingGeometryToken;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theMGFieldToken;
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theMGField;

  const std::string theFitterName;
  edm::ESGetToken<TrajectoryFitter, TrajectoryFitterRecord> theFitterToken;
  std::unique_ptr<TrajectoryFitter> theFitter;

  const std::string theSmootherName;
  edm::ESGetToken<TrajectorySmoother, TrajectoryFitterRecord> theSmootherToken;
  std::unique_ptr<TrajectorySmoother> theSmoother;

  const std::string thePropagatorName;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> thePropagatorToken;
  edm::ESHandle<Propagator> const& propagator() const { return thePropagator; }
  edm::ESHandle<Propagator> thePropagator;

  const std::string theTrackerRecHitBuilderName;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> theTrackerRecHitBuilderToken;
  const TransientTrackingRecHitBuilder* theTrackerRecHitBuilder;
  TkClonerImpl hitCloner;

  const std::string theMuonRecHitBuilderName;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> theMuonRecHitBuilderToken;
  edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;

  const std::string theMTDRecHitBuilderName;
  edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> theMTDRecHitBuilderToken;
  bool theMtdAvailable;
  edm::ESHandle<TransientTrackingRecHitBuilder> theMTDRecHitBuilder;
};
#endif
