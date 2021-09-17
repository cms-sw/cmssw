#ifndef TrackAssociator_TrackDetectorAssociator_h
#define TrackAssociator_TrackDetectorAssociator_h 1

// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      TrackDetectorAssociator
//
/*

 Description: main class of tools to associate a track to calorimeter and muon detectors

*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
//
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"
#include "TrackingTools/TrackAssociator/interface/CachedTrajectory.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "FWCore/Utilities/interface/Visibility.h"

class TrackDetectorAssociator {
public:
  explicit TrackDetectorAssociator();
  ~TrackDetectorAssociator();

  typedef TrackAssociatorParameters AssociatorParameters;
  enum Direction { Any, InsideOut, OutsideIn };

  /// propagate a track across the whole detector and
  /// find associated objects. Association is done in
  /// two modes
  ///  1) an object is associated to a track only if
  ///     crossed by track
  ///  2) an object is associated to a track if it is
  ///     withing an eta-phi cone of some radius with
  ///     respect to a track.
  ///     (the cone origin is at (0,0,0))
  /// Trajectory bending in eta-phi is taking into account
  /// when matching is performed
  ///
  /// associate using FreeTrajectoryState
  TrackDetMatchInfo associate(const edm::Event&,
                              const edm::EventSetup&,
                              const FreeTrajectoryState&,
                              const AssociatorParameters&);
  /// associate using inner and outer most states of a track
  /// in the silicon tracker.
  TrackDetMatchInfo associate(const edm::Event& iEvent,
                              const edm::EventSetup& iSetup,
                              const AssociatorParameters& parameters,
                              const FreeTrajectoryState* innerState,
                              const FreeTrajectoryState* outerState = nullptr);
  /// associate using reco::Track
  TrackDetMatchInfo associate(const edm::Event&,
                              const edm::EventSetup&,
                              const reco::Track&,
                              const AssociatorParameters&,
                              Direction direction = Any);
  /// associate using a simulated track
  TrackDetMatchInfo associate(
      const edm::Event&, const edm::EventSetup&, const SimTrack&, const SimVertex&, const AssociatorParameters&);
  /// associate using 3-momentum, vertex and charge
  TrackDetMatchInfo associate(const edm::Event&,
                              const edm::EventSetup&,
                              const GlobalVector&,
                              const GlobalPoint&,
                              const int,
                              const AssociatorParameters&);
  /// trajector information
  const CachedTrajectory& getCachedTrajector() const { return cachedTrajectory_; }

  /// use a user configured propagator
  void setPropagator(const Propagator*);

  /// use the default propagator
  void useDefaultPropagator();

  /// get FreeTrajectoryState from different track representations
  static FreeTrajectoryState getFreeTrajectoryState(const MagneticField*, const reco::Track&);
  static FreeTrajectoryState getFreeTrajectoryState(const MagneticField*, const SimTrack&, const SimVertex&);
  static FreeTrajectoryState getFreeTrajectoryState(const MagneticField*,
                                                    const GlobalVector&,
                                                    const GlobalPoint&,
                                                    const int);

  static bool crossedIP(const reco::Track& track);

private:
  DetIdAssociator::MapRange getMapRange(const std::pair<float, float>& delta, const float dR) dso_internal;

  void fillEcal(const edm::Event&, TrackDetMatchInfo&, const AssociatorParameters&) dso_internal;

  void fillCaloTowers(const edm::Event&, TrackDetMatchInfo&, const AssociatorParameters&) dso_internal;

  void fillHcal(const edm::Event&, TrackDetMatchInfo&, const AssociatorParameters&) dso_internal;

  void fillHO(const edm::Event&, TrackDetMatchInfo&, const AssociatorParameters&) dso_internal;

  void fillPreshower(const edm::Event& iEvent, TrackDetMatchInfo& info, const AssociatorParameters&) dso_internal;

  void fillMuon(const edm::Event&, TrackDetMatchInfo&, const AssociatorParameters&) dso_internal;

  void fillCaloTruth(const edm::Event&, TrackDetMatchInfo&, const AssociatorParameters&) dso_internal;

  bool addTAMuonSegmentMatch(TAMuonChamberMatch&, const RecSegment*, const AssociatorParameters&) dso_internal;

  void getTAMuonChamberMatches(std::vector<TAMuonChamberMatch>& matches,
                               const AssociatorParameters& parameters) dso_internal;

  void init(const edm::EventSetup&, const AssociatorParameters&) dso_internal;

  math::XYZPoint getPoint(const GlobalPoint& point) dso_internal {
    return math::XYZPoint(point.x(), point.y(), point.z());
  }

  math::XYZPoint getPoint(const LocalPoint& point) dso_internal {
    return math::XYZPoint(point.x(), point.y(), point.z());
  }

  math::XYZVector getVector(const GlobalVector& vec) dso_internal { return math::XYZVector(vec.x(), vec.y(), vec.z()); }

  math::XYZVector getVector(const LocalVector& vec) dso_internal { return math::XYZVector(vec.x(), vec.y(), vec.z()); }

  const Propagator* ivProp_;
  std::unique_ptr<Propagator> defProp_;
  CachedTrajectory cachedTrajectory_;
  bool useDefaultPropagator_;

  const DetIdAssociator* ecalDetIdAssociator_;
  const DetIdAssociator* hcalDetIdAssociator_;
  const DetIdAssociator* hoDetIdAssociator_;
  const DetIdAssociator* caloDetIdAssociator_;
  const DetIdAssociator* muonDetIdAssociator_;
  const DetIdAssociator* preshowerDetIdAssociator_;

  const CaloGeometry* theCaloGeometry_;
  const GlobalTrackingGeometry* theTrackingGeometry_;

  edm::ESWatcher<IdealMagneticFieldRecord> theMagneticFieldWatcher_;
};
#endif
