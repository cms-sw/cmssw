#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

using namespace std;
using namespace edm;

/// Constructor
TrackTransformer::TrackTransformer(const ParameterSet& parameterSet)
    : theRPCInTheFit(parameterSet.getParameter<bool>("RefitRPCHits")),
      theDoPredictionsOnly(parameterSet.getParameter<bool>("DoPredictionsOnly")),
      theRefitDirection(parameterSet.getParameter<string>("RefitDirection")),
      theFitterName(parameterSet.getParameter<string>("Fitter")),
      theSmootherName(parameterSet.getParameter<string>("Smoother")),
      thePropagatorName(parameterSet.getParameter<string>("Propagator")),
      theTrackerRecHitBuilderName(parameterSet.getParameter<string>("TrackerRecHitBuilder")),
      theMuonRecHitBuilderName(parameterSet.getParameter<string>("MuonRecHitBuilder")),
      theMTDRecHitBuilderName(parameterSet.getParameter<string>("MTDRecHitBuilder")) {}

TrackTransformer::TrackTransformer(const ParameterSet& parameterSet, edm::ConsumesCollector& iC)
    : TrackTransformer(parameterSet) {
  theTrackingGeometryToken = iC.esConsumes();
  theMGFieldToken = iC.esConsumes();
  theFitterToken = iC.esConsumes(edm::ESInputTag("", theFitterName));
  theSmootherToken = iC.esConsumes(edm::ESInputTag("", theSmootherName));
  thePropagatorToken = iC.esConsumes(edm::ESInputTag("", thePropagatorName));
  theTrackerRecHitBuilderToken = iC.esConsumes(edm::ESInputTag("", theTrackerRecHitBuilderName));
  theMuonRecHitBuilderToken = iC.esConsumes(edm::ESInputTag("", theMuonRecHitBuilderName));
  theMTDRecHitBuilderToken = iC.esConsumes(edm::ESInputTag("", theMTDRecHitBuilderName));
}

/// Destructor
TrackTransformer::~TrackTransformer() {}

void TrackTransformer::fillPSetDescription(edm::ParameterSetDescription& desc,
                                           bool DoPredictionsOnly,
                                           const std::string& Fitter,
                                           const std::string& Smoother,
                                           const std::string& Propagator,
                                           const std::string& RefitDirection,
                                           bool RefitRPCHits,
                                           const std::string& TrackerRecHitBuilder,
                                           const std::string& MuonRecHitBuilder,
                                           const std::string& MTDRecHitBuilder) {
  desc.add<bool>("DoPredictionsOnly", DoPredictionsOnly);
  desc.add<std::string>("Fitter", Fitter);
  desc.add<std::string>("Smoother", Smoother);
  desc.add<std::string>("Propagator", Propagator);
  desc.add<std::string>("RefitDirection", RefitDirection);
  desc.add<bool>("RefitRPCHits", RefitRPCHits);
  desc.add<std::string>("TrackerRecHitBuilder", TrackerRecHitBuilder);
  desc.add<std::string>("MuonRecHitBuilder", MuonRecHitBuilder);
  desc.add<std::string>("MTDRecHitBuilder", MTDRecHitBuilder);
}

void TrackTransformer::setServices(const EventSetup& setup) {
  const std::string metname = "Reco|TrackingTools|TrackTransformer";

  if (theFitterToken.isInitialized()) {
    theFitter = setup.getData(theFitterToken).clone();
    theSmoother.reset(setup.getData(theSmootherToken).clone());

    thePropagator = setup.getHandle(thePropagatorToken);

    // Global Tracking Geometry
    theTrackingGeometry = setup.getHandle(theTrackingGeometryToken);

    // Magfield Field
    theMGField = setup.getHandle(theMGFieldToken);
  } else {
    edm::ESHandle<TrajectoryFitter> aFitter;
    edm::ESHandle<TrajectorySmoother> aSmoother;
    setup.get<TrajectoryFitter::Record>().get(theFitterName, aFitter);
    setup.get<TrajectoryFitter::Record>().get(theSmootherName, aSmoother);
    theFitter = aFitter->clone();
    theSmoother.reset(aSmoother->clone());

    setup.get<TrackingComponentsRecord>().get(thePropagatorName, thePropagator);

    // Global Tracking Geometry
    setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

    // Magfield Field
    setup.get<IdealMagneticFieldRecord>().get(theMGField);
  }

  // Transient Rechit Builders
  unsigned long long newCacheId_TRH = setup.get<TransientRecHitRecord>().cacheIdentifier();
  if (newCacheId_TRH != theCacheId_TRH) {
    theCacheId_TRH = newCacheId_TRH;
    LogTrace(metname) << "TransientRecHitRecord changed!";
    if (theTrackerRecHitBuilderToken.isInitialized()) {
      theTrackerRecHitBuilder = &setup.getData(theTrackerRecHitBuilderToken);
      theMuonRecHitBuilder = setup.getHandle(theMuonRecHitBuilderToken);
      theMTDRecHitBuilder = setup.getHandle(theMTDRecHitBuilderToken);
    } else {
      edm::ESHandle<TransientTrackingRecHitBuilder> aTrackerRecHitBuilder;
      setup.get<TransientRecHitRecord>().get(theTrackerRecHitBuilderName, aTrackerRecHitBuilder);
      theTrackerRecHitBuilder = aTrackerRecHitBuilder.product();
      setup.get<TransientRecHitRecord>().get(theMuonRecHitBuilderName, theMuonRecHitBuilder);
      setup.get<TransientRecHitRecord>().get(theMTDRecHitBuilderName, theMTDRecHitBuilder);
    }
    theMtdAvailable = theMTDRecHitBuilder.isValid();
    hitCloner = static_cast<TkTransientTrackingRecHitBuilder const*>(theTrackerRecHitBuilder)->cloner();
  }
  theFitter->setHitCloner(&hitCloner);
  theSmoother->setHitCloner(&hitCloner);
}

vector<Trajectory> TrackTransformer::transform(const reco::TrackRef& track) const { return transform(*track); }

TransientTrackingRecHit::ConstRecHitContainer TrackTransformer::getTransientRecHits(
    const reco::TransientTrack& track) const {
  TransientTrackingRecHit::ConstRecHitContainer result;
  auto tkbuilder = static_cast<TkTransientTrackingRecHitBuilder const*>(theTrackerRecHitBuilder);

  for (auto hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit) {
    if ((*hit)->isValid()) {
      if ((*hit)->geographicalId().det() == DetId::Tracker) {
        result.emplace_back((**hit).cloneForFit(*tkbuilder->geometry()->idToDet((**hit).geographicalId())));
      } else if ((*hit)->geographicalId().det() == DetId::Muon) {
        if ((*hit)->geographicalId().subdetId() == 3 && !theRPCInTheFit) {
          LogTrace("Reco|TrackingTools|TrackTransformer") << "RPC Rec Hit discarged";
          continue;
        }
        result.push_back(theMuonRecHitBuilder->build(&**hit));
      } else if ((*hit)->geographicalId().det() == DetId::Forward && (*hit)->geographicalId().subdetId() == FastTime) {
        if (theMtdAvailable)
          result.push_back(theMTDRecHitBuilder->build(&**hit));
        else
          throw cms::Exception("TrackTransformer") << "MTD hit encountered but MTD not available!";
      }
    }
  }

  return result;
}

// FIXME: check this method!
RefitDirection::GeometricalDirection TrackTransformer::checkRecHitsOrdering(
    TransientTrackingRecHit::ConstRecHitContainer const& recHits) const {
  if (!recHits.empty()) {
    GlobalPoint first = trackingGeometry()->idToDet(recHits.front()->geographicalId())->position();
    GlobalPoint last = trackingGeometry()->idToDet(recHits.back()->geographicalId())->position();

    // maybe perp2?
    auto rFirst = first.mag2();
    auto rLast = last.mag2();
    if (rFirst < rLast)
      return RefitDirection::insideOut;
    if (rFirst > rLast)
      return RefitDirection::outsideIn;
  }
  LogDebug("Reco|TrackingTools|TrackTransformer") << "Impossible to determine the rechits order" << endl;
  return RefitDirection::undetermined;
}

/// Convert Tracks into Trajectories
vector<Trajectory> TrackTransformer::transform(const reco::Track& newTrack) const {
  const std::string metname = "Reco|TrackingTools|TrackTransformer";

  reco::TransientTrack track(newTrack, magneticField(), trackingGeometry());

  auto recHitsForReFit = getTransientRecHits(track);
  return transform(track, recHitsForReFit);
}

/// Convert Tracks into Trajectories with a given set of hits
vector<Trajectory> TrackTransformer::transform(const reco::TransientTrack& track,
                                               TransientTrackingRecHit::ConstRecHitContainer& recHitsForReFit) const {
  const std::string metname = "Reco|TrackingTools|TrackTransformer";

  if (recHitsForReFit.size() < 2)
    return vector<Trajectory>();

  // 8 cases are foreseen:
  // [RH = rec hit order, P = momentum dir, FD = fit direction. IO/OI = inside-out/outside-in, AM/OM = along momentum/opposite to momentum]
  // (1) RH IO | P IO | FD AM  ---> Start from IN
  // (2) RH IO | P IO | FD OM  ---> Reverse RH and start from OUT
  // (3) RH IO | P OI | FD AM  ---> Reverse RH and start from IN
  // (4) RH IO | P OI | FD OM  ---> Start from OUT
  // (5) RH OI | P IO | FD AM  ---> Reverse RH and start from IN
  // (6) RH OI | P IO | FD OM  ---> Start from OUT
  // (7) RH OI | P OI | FD AM  ---> Start from IN
  // (8) RH OI | P OI | FD OM  ---> Reverse RH and start from OUT
  //
  // *** Rules: ***
  // -A- If RH-FD agree (IO-AM,OI-OM) do not reverse the RH
  // -B- If FD along momentum start from innermost state, otherwise use outermost

  // Other special cases can be handled:
  // (1 bis) RH IO | P IO | GFD IO => FD AM  ---> Start from IN
  // (2 bis) RH IO | P IO | GFD OI => FD OM  ---> Reverse RH and start from OUT
  // (3 bis) RH IO | P OI | GFD OI => FD AM  ---> Reverse RH and start from OUT
  // (4 bis) RH IO | P OI | GFD IO => FD OM  ---> Start from IN
  // (5 bis) RH OI | P IO | GFD IO => FD AM  ---> Reverse RH and start from IN
  // (6 bis) RH OI | P IO | GFD OI => FD OM  ---> Start from OUT
  // (7 bis) RH OI | P OI | GFD OI => FD AM  ---> Start from OUT
  // (8 bis) RH OI | P OI | GFD IO => FD OM  ---> Reverse RH and start from IN
  //
  // *** Additional rule: ***
  // -A0- If P and GFD agree, then FD is AM otherwise is OM
  // -A00- rechit must be ordered as GFD in order to handle the case of cosmics
  // -B0- The starting state is decided by GFD

  // Determine the RH order
  RefitDirection::GeometricalDirection recHitsOrder =
      checkRecHitsOrdering(recHitsForReFit);  // FIXME change nome of the *type*  --> RecHit order!
  LogTrace(metname) << "RH order (0-insideOut, 1-outsideIn): " << recHitsOrder;

  PropagationDirection propagationDirection = theRefitDirection.propagationDirection();

  // Apply rule -A0-
  if (propagationDirection == anyDirection) {
    GlobalVector momentum = track.innermostMeasurementState().globalMomentum();
    GlobalVector position = track.innermostMeasurementState().globalPosition() - GlobalPoint(0, 0, 0);
    RefitDirection::GeometricalDirection p = (momentum.x() * position.x() > 0 || momentum.y() * position.y() > 0)
                                                 ? RefitDirection::insideOut
                                                 : RefitDirection::outsideIn;

    propagationDirection = p == theRefitDirection.geometricalDirection() ? alongMomentum : oppositeToMomentum;
    LogTrace(metname) << "P  (0-insideOut, 1-outsideIn): " << p;
    LogTrace(metname) << "FD (0-OM, 1-AM, 2-ANY): " << propagationDirection;
  }
  // -A0-

  // Apply rule -A-
  if (theRefitDirection.propagationDirection() != anyDirection) {
    if ((recHitsOrder == RefitDirection::insideOut && propagationDirection == oppositeToMomentum) ||
        (recHitsOrder == RefitDirection::outsideIn && propagationDirection == alongMomentum))
      reverse(recHitsForReFit.begin(), recHitsForReFit.end());
  }
  // -A-
  // Apply rule -A00-
  else {
    // reorder the rechit as defined in theRefitDirection.geometricalDirection();
    if (theRefitDirection.geometricalDirection() != recHitsOrder)
      reverse(recHitsForReFit.begin(), recHitsForReFit.end());
  }
  // -A00-

  // Apply rule -B-
  TrajectoryStateOnSurface firstTSOS = track.innermostMeasurementState();
  unsigned int innerId = track.track().innerDetId();
  if (theRefitDirection.propagationDirection() != anyDirection) {
    if (propagationDirection == oppositeToMomentum) {
      innerId = track.track().outerDetId();
      firstTSOS = track.outermostMeasurementState();
    }
  } else {  // if(theRefitDirection.propagationDirection() == anyDirection)
    // Apply rule -B0-
    if (theRefitDirection.geometricalDirection() == RefitDirection::outsideIn) {
      innerId = track.track().outerDetId();
      firstTSOS = track.outermostMeasurementState();
    }
    // -B0-
  }
  // -B-

  if (!firstTSOS.isValid()) {
    LogTrace(metname) << "Error wrong initial state!" << endl;
    return vector<Trajectory>();
  }

  TrajectorySeed seed({}, {}, propagationDirection);

  if (recHitsForReFit.front()->geographicalId() != DetId(innerId)) {
    LogTrace(metname) << "Propagation occured" << endl;
    firstTSOS = propagator()->propagate(firstTSOS, recHitsForReFit.front()->det()->surface());
    if (!firstTSOS.isValid()) {
      LogTrace(metname) << "Propagation error!" << endl;
      return vector<Trajectory>();
    }
  }

  if (theDoPredictionsOnly) {
    Trajectory aTraj(seed, propagationDirection);
    TrajectoryStateOnSurface predTSOS = firstTSOS;
    for (auto const& hit : recHitsForReFit) {
      predTSOS = propagator()->propagate(predTSOS, hit->det()->surface());
      if (predTSOS.isValid())
        aTraj.push(TrajectoryMeasurement(predTSOS, hit));
    }
    return vector<Trajectory>(1, aTraj);
  }

  auto const& trajectories = theFitter->fit(seed, recHitsForReFit, firstTSOS);

  if (trajectories.empty()) {
    LogTrace(metname) << "No Track refitted!" << endl;
    return trajectories;
  }

  auto const& trajectoryBW = trajectories.front();

  auto const& trajectoriesSM = theSmoother->trajectories(trajectoryBW);

  if (trajectoriesSM.empty()) {
    LogTrace(metname) << "No Track smoothed!" << endl;
  }

  return trajectoriesSM;
}
