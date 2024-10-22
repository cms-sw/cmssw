// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      TrackDetectorAssociator
//
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
//
//

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrackAssociator/interface/DetIdInfo.h"
#include "TrackingTools/Records/interface/DetIdAssociatorRecord.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0SegmentCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"

// calorimeter and muon infos
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include <stack>
#include <set>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "Math/VectorUtil.h"
#include <algorithm>

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

using namespace reco;

TrackDetectorAssociator::TrackDetectorAssociator() {
  ivProp_ = nullptr;
  useDefaultPropagator_ = false;
}

TrackDetectorAssociator::~TrackDetectorAssociator() = default;

void TrackDetectorAssociator::setPropagator(const Propagator* ptr) {
  ivProp_ = ptr;
  cachedTrajectory_.setPropagator(ivProp_);
}

void TrackDetectorAssociator::useDefaultPropagator() { useDefaultPropagator_ = true; }

void TrackDetectorAssociator::init(const edm::EventSetup& iSetup, const AssociatorParameters& parameters) {
  // access the calorimeter geometry
  theCaloGeometry_ = &iSetup.getData(parameters.theCaloGeometryToken);

  // get the tracking Geometry
  theTrackingGeometry_ = &iSetup.getData(parameters.theTrackingGeometryToken);

  if (useDefaultPropagator_ && (!defProp_ || theMagneticFieldWatcher_.check(iSetup))) {
    // setup propagator
    const MagneticField* bField = &iSetup.getData(parameters.bFieldToken);

    auto prop = std::make_unique<SteppingHelixPropagator>(bField, anyDirection);
    prop->setMaterialMode(false);
    prop->applyRadX0Correction(true);
    // prop->setDebug(true); // tmp
    defProp_ = std::move(prop);
    setPropagator(defProp_.get());
  }

  ecalDetIdAssociator_ = &iSetup.getData(parameters.ecalDetIdAssociatorToken);
  hcalDetIdAssociator_ = &iSetup.getData(parameters.hcalDetIdAssociatorToken);
  hoDetIdAssociator_ = &iSetup.getData(parameters.hoDetIdAssociatorToken);
  caloDetIdAssociator_ = &iSetup.getData(parameters.caloDetIdAssociatorToken);
  muonDetIdAssociator_ = &iSetup.getData(parameters.muonDetIdAssociatorToken);
  preshowerDetIdAssociator_ = &iSetup.getData(parameters.preshowerDetIdAssociatorToken);
}

TrackDetMatchInfo TrackDetectorAssociator::associate(const edm::Event& iEvent,
                                                     const edm::EventSetup& iSetup,
                                                     const FreeTrajectoryState& fts,
                                                     const AssociatorParameters& parameters) {
  return associate(iEvent, iSetup, parameters, &fts);
}

TrackDetMatchInfo TrackDetectorAssociator::associate(const edm::Event& iEvent,
                                                     const edm::EventSetup& iSetup,
                                                     const AssociatorParameters& parameters,
                                                     const FreeTrajectoryState* innerState,
                                                     const FreeTrajectoryState* outerState) {
  TrackDetMatchInfo info;
  if (!parameters.useEcal && !parameters.useCalo && !parameters.useHcal && !parameters.useHO && !parameters.useMuon &&
      !parameters.usePreshower)
    throw cms::Exception("ConfigurationError")
        << "Configuration error! No subdetector was selected for the track association.";

  SteppingHelixStateInfo trackOrigin(*innerState);
  info.stateAtIP = *innerState;
  cachedTrajectory_.setStateAtIP(trackOrigin);

  init(iSetup, parameters);
  // get track trajectory
  // ECAL points (EB+EE)
  // If the phi angle between a track entrance and exit points is more
  // than 2 crystals, it is possible that the track will cross 3 crystals
  // and therefore one has to check at least 3 points along the track
  // trajectory inside ECAL. In order to have a chance to cross 4 crystalls
  // in the barrel, a track should have P_t as low as 3 GeV or smaller
  // If it's necessary, number of points along trajectory can be increased

  info.setCaloGeometry(theCaloGeometry_);

  cachedTrajectory_.reset_trajectory();
  // estimate propagation outer boundaries based on
  // requested sub-detector information. For now limit
  // propagation region only if muon matching is not
  // requested.
  double HOmaxR = hoDetIdAssociator_->volume().maxR();
  double HOmaxZ = hoDetIdAssociator_->volume().maxZ();
  double minR = ecalDetIdAssociator_->volume().minR();
  double minZ = preshowerDetIdAssociator_->volume().minZ();
  cachedTrajectory_.setMaxHORadius(HOmaxR);
  cachedTrajectory_.setMaxHOLength(HOmaxZ * 2.);
  cachedTrajectory_.setMinDetectorRadius(minR);
  cachedTrajectory_.setMinDetectorLength(minZ * 2.);

  if (parameters.useMuon) {
    double maxR = muonDetIdAssociator_->volume().maxR();
    double maxZ = muonDetIdAssociator_->volume().maxZ();
    cachedTrajectory_.setMaxDetectorRadius(maxR);
    cachedTrajectory_.setMaxDetectorLength(maxZ * 2.);
  } else {
    cachedTrajectory_.setMaxDetectorRadius(HOmaxR);
    cachedTrajectory_.setMaxDetectorLength(HOmaxZ * 2.);
  }

  // If track extras exist and outerState is before HO maximum, then use outerState
  if (outerState) {
    if (outerState->position().perp() < HOmaxR && std::abs(outerState->position().z()) < HOmaxZ) {
      LogTrace("TrackAssociator") << "Using outerState as trackOrigin at Rho=" << outerState->position().perp()
                                  << "  Z=" << outerState->position().z() << "\n";
      trackOrigin = SteppingHelixStateInfo(*outerState);
    } else if (innerState) {
      LogTrace("TrackAssociator") << "Using innerState as trackOrigin at Rho=" << innerState->position().perp()
                                  << "  Z=" << innerState->position().z() << "\n";
      trackOrigin = SteppingHelixStateInfo(*innerState);
    }
  }

  if (trackOrigin.momentum().mag() == 0)
    return info;
  if (edm::isNotFinite(trackOrigin.momentum().x()) or edm::isNotFinite(trackOrigin.momentum().y()) or
      edm::isNotFinite(trackOrigin.momentum().z()))
    return info;
  if (!cachedTrajectory_.propagateAll(trackOrigin))
    return info;

  // get trajectory in calorimeters
  cachedTrajectory_.findEcalTrajectory(ecalDetIdAssociator_->volume());
  cachedTrajectory_.findHcalTrajectory(hcalDetIdAssociator_->volume());
  cachedTrajectory_.findHOTrajectory(hoDetIdAssociator_->volume());
  cachedTrajectory_.findPreshowerTrajectory(preshowerDetIdAssociator_->volume());

  info.trkGlobPosAtEcal = getPoint(cachedTrajectory_.getStateAtEcal().position());
  info.trkGlobPosAtHcal = getPoint(cachedTrajectory_.getStateAtHcal().position());
  info.trkGlobPosAtHO = getPoint(cachedTrajectory_.getStateAtHO().position());

  info.trkMomAtEcal = cachedTrajectory_.getStateAtEcal().momentum();
  info.trkMomAtHcal = cachedTrajectory_.getStateAtHcal().momentum();
  info.trkMomAtHO = cachedTrajectory_.getStateAtHO().momentum();

  if (parameters.useEcal)
    fillEcal(iEvent, info, parameters);
  if (parameters.useCalo)
    fillCaloTowers(iEvent, info, parameters);
  if (parameters.useHcal)
    fillHcal(iEvent, info, parameters);
  if (parameters.useHO)
    fillHO(iEvent, info, parameters);
  if (parameters.usePreshower)
    fillPreshower(iEvent, info, parameters);
  if (parameters.useMuon)
    fillMuon(iEvent, info, parameters);
  if (parameters.truthMatch)
    fillCaloTruth(iEvent, info, parameters);

  return info;
}

void TrackDetectorAssociator::fillEcal(const edm::Event& iEvent,
                                       TrackDetMatchInfo& info,
                                       const AssociatorParameters& parameters) {
  const std::vector<SteppingHelixStateInfo>& trajectoryStates = cachedTrajectory_.getEcalTrajectory();

  for (std::vector<SteppingHelixStateInfo>::const_iterator itr = trajectoryStates.begin();
       itr != trajectoryStates.end();
       itr++)
    LogTrace("TrackAssociator") << "ECAL trajectory point (rho, z, phi): " << itr->position().perp() << ", "
                                << itr->position().z() << ", " << itr->position().phi();

  std::vector<GlobalPoint> coreTrajectory;
  for (std::vector<SteppingHelixStateInfo>::const_iterator itr = trajectoryStates.begin();
       itr != trajectoryStates.end();
       itr++)
    coreTrajectory.push_back(itr->position());

  if (coreTrajectory.empty()) {
    LogTrace("TrackAssociator") << "ECAL track trajectory is empty; moving on\n";
    info.isGoodEcal = false;
    return;
  }
  info.isGoodEcal = true;

  // Find ECAL crystals
  edm::Handle<EBRecHitCollection> EBRecHits;
  iEvent.getByToken(parameters.EBRecHitsToken, EBRecHits);
  if (!EBRecHits.isValid())
    throw cms::Exception("FatalError") << "Unable to find EBRecHitCollection in the event!\n";

  edm::Handle<EERecHitCollection> EERecHits;
  iEvent.getByToken(parameters.EERecHitsToken, EERecHits);
  if (!EERecHits.isValid())
    throw cms::Exception("FatalError") << "Unable to find EERecHitCollection in event!\n";

  std::set<DetId> ecalIdsInRegion;
  if (parameters.accountForTrajectoryChangeCalo) {
    // get trajectory change with respect to initial state
    DetIdAssociator::MapRange mapRange =
        getMapRange(cachedTrajectory_.trajectoryDelta(CachedTrajectory::IpToEcal), parameters.dREcalPreselection);
    ecalIdsInRegion = ecalDetIdAssociator_->getDetIdsCloseToAPoint(coreTrajectory[0], mapRange);
  } else
    ecalIdsInRegion = ecalDetIdAssociator_->getDetIdsCloseToAPoint(coreTrajectory[0], parameters.dREcalPreselection);
  LogTrace("TrackAssociator") << "ECAL hits in the region: " << ecalIdsInRegion.size();
  if (parameters.dREcalPreselection > parameters.dREcal)
    ecalIdsInRegion = ecalDetIdAssociator_->getDetIdsInACone(ecalIdsInRegion, coreTrajectory, parameters.dREcal);
  LogTrace("TrackAssociator") << "ECAL hits in the cone: " << ecalIdsInRegion.size();
  info.crossedEcalIds = ecalDetIdAssociator_->getCrossedDetIds(ecalIdsInRegion, coreTrajectory);
  const std::vector<DetId>& crossedEcalIds = info.crossedEcalIds;
  LogTrace("TrackAssociator") << "ECAL crossed hits " << crossedEcalIds.size();

  // add EcalRecHits
  for (std::vector<DetId>::const_iterator itr = crossedEcalIds.begin(); itr != crossedEcalIds.end(); itr++) {
    std::vector<EcalRecHit>::const_iterator ebHit = (*EBRecHits).find(*itr);
    std::vector<EcalRecHit>::const_iterator eeHit = (*EERecHits).find(*itr);
    if (ebHit != (*EBRecHits).end())
      info.crossedEcalRecHits.push_back(&*ebHit);
    else if (eeHit != (*EERecHits).end())
      info.crossedEcalRecHits.push_back(&*eeHit);
    else
      LogTrace("TrackAssociator") << "Crossed EcalRecHit is not found for DetId: " << itr->rawId();
  }
  for (std::set<DetId>::const_iterator itr = ecalIdsInRegion.begin(); itr != ecalIdsInRegion.end(); itr++) {
    std::vector<EcalRecHit>::const_iterator ebHit = (*EBRecHits).find(*itr);
    std::vector<EcalRecHit>::const_iterator eeHit = (*EERecHits).find(*itr);
    if (ebHit != (*EBRecHits).end())
      info.ecalRecHits.push_back(&*ebHit);
    else if (eeHit != (*EERecHits).end())
      info.ecalRecHits.push_back(&*eeHit);
    else
      LogTrace("TrackAssociator") << "EcalRecHit from the cone is not found for DetId: " << itr->rawId();
  }
}

void TrackDetectorAssociator::fillCaloTowers(const edm::Event& iEvent,
                                             TrackDetMatchInfo& info,
                                             const AssociatorParameters& parameters) {
  // use ECAL and HCAL trajectories to match a tower. (HO isn't used for matching).
  std::vector<GlobalPoint> trajectory;
  const std::vector<SteppingHelixStateInfo>& ecalTrajectoryStates = cachedTrajectory_.getEcalTrajectory();
  const std::vector<SteppingHelixStateInfo>& hcalTrajectoryStates = cachedTrajectory_.getHcalTrajectory();
  for (std::vector<SteppingHelixStateInfo>::const_iterator itr = ecalTrajectoryStates.begin();
       itr != ecalTrajectoryStates.end();
       itr++)
    trajectory.push_back(itr->position());
  for (std::vector<SteppingHelixStateInfo>::const_iterator itr = hcalTrajectoryStates.begin();
       itr != hcalTrajectoryStates.end();
       itr++)
    trajectory.push_back(itr->position());

  if (trajectory.empty()) {
    LogTrace("TrackAssociator") << "HCAL trajectory is empty; moving on\n";
    info.isGoodCalo = false;
    return;
  }
  info.isGoodCalo = true;

  // find crossed CaloTowers
  edm::Handle<CaloTowerCollection> caloTowers;
  iEvent.getByToken(parameters.caloTowersToken, caloTowers);
  if (!caloTowers.isValid())
    throw cms::Exception("FatalError") << "Unable to find CaloTowers in event!\n";

  std::set<DetId> caloTowerIdsInRegion;
  if (parameters.accountForTrajectoryChangeCalo) {
    // get trajectory change with respect to initial state
    DetIdAssociator::MapRange mapRange =
        getMapRange(cachedTrajectory_.trajectoryDelta(CachedTrajectory::IpToHcal), parameters.dRHcalPreselection);
    caloTowerIdsInRegion = caloDetIdAssociator_->getDetIdsCloseToAPoint(trajectory[0], mapRange);
  } else
    caloTowerIdsInRegion = caloDetIdAssociator_->getDetIdsCloseToAPoint(trajectory[0], parameters.dRHcalPreselection);

  LogTrace("TrackAssociator") << "Towers in the region: " << caloTowerIdsInRegion.size();

  auto caloTowerIdsInAConeBegin = caloTowerIdsInRegion.begin();
  auto caloTowerIdsInAConeEnd = caloTowerIdsInRegion.end();
  std::set<DetId> caloTowerIdsInAConeTmp;
  if (!caloDetIdAssociator_->selectAllInACone(parameters.dRHcal)) {
    caloTowerIdsInAConeTmp =
        caloDetIdAssociator_->getDetIdsInACone(caloTowerIdsInRegion, trajectory, parameters.dRHcal);
    caloTowerIdsInAConeBegin = caloTowerIdsInAConeTmp.begin();
    caloTowerIdsInAConeEnd = caloTowerIdsInAConeTmp.end();
  }
  LogTrace("TrackAssociator") << "Towers in the cone: "
                              << std::distance(caloTowerIdsInAConeBegin, caloTowerIdsInAConeEnd);

  info.crossedTowerIds = caloDetIdAssociator_->getCrossedDetIds(caloTowerIdsInRegion, trajectory);
  const std::vector<DetId>& crossedCaloTowerIds = info.crossedTowerIds;
  LogTrace("TrackAssociator") << "Towers crossed: " << crossedCaloTowerIds.size();

  // add CaloTowers
  for (std::vector<DetId>::const_iterator itr = crossedCaloTowerIds.begin(); itr != crossedCaloTowerIds.end(); itr++) {
    CaloTowerCollection::const_iterator tower = (*caloTowers).find(*itr);
    if (tower != (*caloTowers).end())
      info.crossedTowers.push_back(&*tower);
    else
      LogTrace("TrackAssociator") << "Crossed CaloTower is not found for DetId: " << (*itr).rawId();
  }

  for (std::set<DetId>::const_iterator itr = caloTowerIdsInAConeBegin; itr != caloTowerIdsInAConeEnd; itr++) {
    CaloTowerCollection::const_iterator tower = (*caloTowers).find(*itr);
    if (tower != (*caloTowers).end())
      info.towers.push_back(&*tower);
    else
      LogTrace("TrackAssociator") << "CaloTower from the cone is not found for DetId: " << (*itr).rawId();
  }
}

void TrackDetectorAssociator::fillPreshower(const edm::Event& iEvent,
                                            TrackDetMatchInfo& info,
                                            const AssociatorParameters& parameters) {
  std::vector<GlobalPoint> trajectory;
  const std::vector<SteppingHelixStateInfo>& trajectoryStates = cachedTrajectory_.getPreshowerTrajectory();
  for (std::vector<SteppingHelixStateInfo>::const_iterator itr = trajectoryStates.begin();
       itr != trajectoryStates.end();
       itr++)
    trajectory.push_back(itr->position());

  if (trajectory.empty()) {
    LogTrace("TrackAssociator") << "Preshower trajectory is empty; moving on\n";
    return;
  }

  std::set<DetId> idsInRegion =
      preshowerDetIdAssociator_->getDetIdsCloseToAPoint(trajectory[0], parameters.dRPreshowerPreselection);

  LogTrace("TrackAssociator") << "Number of Preshower Ids in the region: " << idsInRegion.size();
  info.crossedPreshowerIds = preshowerDetIdAssociator_->getCrossedDetIds(idsInRegion, trajectory);
  LogTrace("TrackAssociator") << "Number of Preshower Ids in crossed: " << info.crossedPreshowerIds.size();
}

void TrackDetectorAssociator::fillHcal(const edm::Event& iEvent,
                                       TrackDetMatchInfo& info,
                                       const AssociatorParameters& parameters) {
  const std::vector<SteppingHelixStateInfo>& trajectoryStates = cachedTrajectory_.getHcalTrajectory();

  std::vector<GlobalPoint> coreTrajectory;
  for (std::vector<SteppingHelixStateInfo>::const_iterator itr = trajectoryStates.begin();
       itr != trajectoryStates.end();
       itr++)
    coreTrajectory.push_back(itr->position());

  if (coreTrajectory.empty()) {
    LogTrace("TrackAssociator") << "HCAL trajectory is empty; moving on\n";
    info.isGoodHcal = false;
    return;
  }
  info.isGoodHcal = true;

  // find crossed Hcals
  edm::Handle<HBHERecHitCollection> collection;
  iEvent.getByToken(parameters.HBHEcollToken, collection);
  if (!collection.isValid())
    throw cms::Exception("FatalError") << "Unable to find HBHERecHits in event!\n";

  std::set<DetId> idsInRegion;
  if (parameters.accountForTrajectoryChangeCalo) {
    // get trajectory change with respect to initial state
    DetIdAssociator::MapRange mapRange =
        getMapRange(cachedTrajectory_.trajectoryDelta(CachedTrajectory::IpToHcal), parameters.dRHcalPreselection);
    idsInRegion = hcalDetIdAssociator_->getDetIdsCloseToAPoint(coreTrajectory[0], mapRange);
  } else
    idsInRegion = hcalDetIdAssociator_->getDetIdsCloseToAPoint(coreTrajectory[0], parameters.dRHcalPreselection);

  LogTrace("TrackAssociator") << "HCAL hits in the region: " << idsInRegion.size() << "\n"
                              << DetIdInfo::info(idsInRegion, nullptr);

  auto idsInAConeBegin = idsInRegion.begin();
  auto idsInAConeEnd = idsInRegion.end();
  std::set<DetId> idsInAConeTmp;
  if (!hcalDetIdAssociator_->selectAllInACone(parameters.dRHcal)) {
    idsInAConeTmp = hcalDetIdAssociator_->getDetIdsInACone(idsInRegion, coreTrajectory, parameters.dRHcal);
    idsInAConeBegin = idsInAConeTmp.begin();
    idsInAConeEnd = idsInAConeTmp.end();
  }
  LogTrace("TrackAssociator") << "HCAL hits in the cone: " << std::distance(idsInAConeBegin, idsInAConeEnd) << "\n"
                              << DetIdInfo::info(std::set<DetId>(idsInAConeBegin, idsInAConeEnd), nullptr);
  info.crossedHcalIds = hcalDetIdAssociator_->getCrossedDetIds(idsInRegion, coreTrajectory);
  const std::vector<DetId>& crossedIds = info.crossedHcalIds;
  LogTrace("TrackAssociator") << "HCAL hits crossed: " << crossedIds.size() << "\n"
                              << DetIdInfo::info(crossedIds, nullptr);

  // add Hcal
  for (std::vector<DetId>::const_iterator itr = crossedIds.begin(); itr != crossedIds.end(); itr++) {
    HBHERecHitCollection::const_iterator hit = (*collection).find(*itr);
    if (hit != (*collection).end())
      info.crossedHcalRecHits.push_back(&*hit);
    else
      LogTrace("TrackAssociator") << "Crossed HBHERecHit is not found for DetId: " << itr->rawId();
  }
  for (std::set<DetId>::const_iterator itr = idsInAConeBegin; itr != idsInAConeEnd; itr++) {
    HBHERecHitCollection::const_iterator hit = (*collection).find(*itr);
    if (hit != (*collection).end())
      info.hcalRecHits.push_back(&*hit);
    else
      LogTrace("TrackAssociator") << "HBHERecHit from the cone is not found for DetId: " << itr->rawId();
  }
}

void TrackDetectorAssociator::fillHO(const edm::Event& iEvent,
                                     TrackDetMatchInfo& info,
                                     const AssociatorParameters& parameters) {
  const std::vector<SteppingHelixStateInfo>& trajectoryStates = cachedTrajectory_.getHOTrajectory();

  std::vector<GlobalPoint> coreTrajectory;
  for (std::vector<SteppingHelixStateInfo>::const_iterator itr = trajectoryStates.begin();
       itr != trajectoryStates.end();
       itr++)
    coreTrajectory.push_back(itr->position());

  if (coreTrajectory.empty()) {
    LogTrace("TrackAssociator") << "HO trajectory is empty; moving on\n";
    info.isGoodHO = false;
    return;
  }
  info.isGoodHO = true;

  // find crossed HOs
  edm::Handle<HORecHitCollection> collection;
  iEvent.getByToken(parameters.HOcollToken, collection);
  if (!collection.isValid())
    throw cms::Exception("FatalError") << "Unable to find HORecHits in event!\n";

  std::set<DetId> idsInRegion;
  if (parameters.accountForTrajectoryChangeCalo) {
    // get trajectory change with respect to initial state
    DetIdAssociator::MapRange mapRange =
        getMapRange(cachedTrajectory_.trajectoryDelta(CachedTrajectory::IpToHO), parameters.dRHcalPreselection);
    idsInRegion = hoDetIdAssociator_->getDetIdsCloseToAPoint(coreTrajectory[0], mapRange);
  } else
    idsInRegion = hoDetIdAssociator_->getDetIdsCloseToAPoint(coreTrajectory[0], parameters.dRHcalPreselection);

  LogTrace("TrackAssociator") << "idsInRegion.size(): " << idsInRegion.size();

  auto idsInAConeBegin = idsInRegion.begin();
  auto idsInAConeEnd = idsInRegion.end();
  std::set<DetId> idsInAConeTmp;
  if (!hoDetIdAssociator_->selectAllInACone(parameters.dRHcal)) {
    idsInAConeTmp = hoDetIdAssociator_->getDetIdsInACone(idsInRegion, coreTrajectory, parameters.dRHcal);
    idsInAConeBegin = idsInAConeTmp.begin();
    idsInAConeEnd = idsInAConeTmp.end();
  }
  LogTrace("TrackAssociator") << "idsInACone.size(): " << std::distance(idsInAConeBegin, idsInAConeEnd);
  info.crossedHOIds = hoDetIdAssociator_->getCrossedDetIds(idsInRegion, coreTrajectory);
  const std::vector<DetId>& crossedIds = info.crossedHOIds;

  // add HO
  for (std::vector<DetId>::const_iterator itr = crossedIds.begin(); itr != crossedIds.end(); itr++) {
    HORecHitCollection::const_iterator hit = (*collection).find(*itr);
    if (hit != (*collection).end())
      info.crossedHORecHits.push_back(&*hit);
    else
      LogTrace("TrackAssociator") << "Crossed HORecHit is not found for DetId: " << itr->rawId();
  }

  for (std::set<DetId>::const_iterator itr = idsInAConeBegin; itr != idsInAConeEnd; itr++) {
    HORecHitCollection::const_iterator hit = (*collection).find(*itr);
    if (hit != (*collection).end())
      info.hoRecHits.push_back(&*hit);
    else
      LogTrace("TrackAssociator") << "HORecHit from the cone is not found for DetId: " << itr->rawId();
  }
}

FreeTrajectoryState TrackDetectorAssociator::getFreeTrajectoryState(const MagneticField* bField,
                                                                    const SimTrack& track,
                                                                    const SimVertex& vertex) {
  GlobalVector vector(track.momentum().x(), track.momentum().y(), track.momentum().z());
  GlobalPoint point(vertex.position().x(), vertex.position().y(), vertex.position().z());

  int charge = track.type() > 0 ? -1 : 1;  // lepton convention
  if (abs(track.type()) == 211 ||          // pion
      abs(track.type()) == 321 ||          // kaon
      abs(track.type()) == 2212)
    charge = track.type() < 0 ? -1 : 1;
  return getFreeTrajectoryState(bField, vector, point, charge);
}

FreeTrajectoryState TrackDetectorAssociator::getFreeTrajectoryState(const MagneticField* bField,
                                                                    const GlobalVector& momentum,
                                                                    const GlobalPoint& vertex,
                                                                    const int charge) {
  GlobalTrajectoryParameters tPars(vertex, momentum, charge, bField);

  ROOT::Math::SMatrixIdentity id;
  AlgebraicSymMatrix66 covT(id);
  covT *= 1e-6;  // initialize to sigma=1e-3
  CartesianTrajectoryError tCov(covT);

  return FreeTrajectoryState(tPars, tCov);
}

FreeTrajectoryState TrackDetectorAssociator::getFreeTrajectoryState(const MagneticField* bField,
                                                                    const reco::Track& track) {
  GlobalVector vector(track.momentum().x(), track.momentum().y(), track.momentum().z());

  GlobalPoint point(track.vertex().x(), track.vertex().y(), track.vertex().z());

  GlobalTrajectoryParameters tPars(point, vector, track.charge(), bField);

  // FIX THIS !!!
  // need to convert from perigee to global or helix (curvilinear) frame
  // for now just an arbitrary matrix.
  ROOT::Math::SMatrixIdentity id;
  AlgebraicSymMatrix66 covT(id);
  covT *= 1e-6;  // initialize to sigma=1e-3
  CartesianTrajectoryError tCov(covT);

  return FreeTrajectoryState(tPars, tCov);
}

DetIdAssociator::MapRange TrackDetectorAssociator::getMapRange(const std::pair<float, float>& delta, const float dR) {
  DetIdAssociator::MapRange mapRange;
  mapRange.dThetaPlus = dR;
  mapRange.dThetaMinus = dR;
  mapRange.dPhiPlus = dR;
  mapRange.dPhiMinus = dR;
  if (delta.first > 0)
    mapRange.dThetaPlus += delta.first;
  else
    mapRange.dThetaMinus += std::abs(delta.first);
  if (delta.second > 0)
    mapRange.dPhiPlus += delta.second;
  else
    mapRange.dPhiMinus += std::abs(delta.second);
  LogTrace("TrackAssociator") << "Selection range: (dThetaPlus, dThetaMinus, dPhiPlus, dPhiMinus, dRPreselection): "
                              << mapRange.dThetaPlus << ", " << mapRange.dThetaMinus << ", " << mapRange.dPhiPlus
                              << ", " << mapRange.dPhiMinus << ", " << dR;
  return mapRange;
}

void TrackDetectorAssociator::getTAMuonChamberMatches(std::vector<TAMuonChamberMatch>& matches,
                                                      const AssociatorParameters& parameters,
                                                      std::set<DetId> occupancy) {
  // Strategy:
  //    Propagate through the whole detector, estimate change in eta and phi
  //    along the trajectory, add this to dRMuon and find DetIds around this
  //    direction using the map. Then propagate fast to each surface and apply
  //    final matching criteria.

  // get the direction first
  SteppingHelixStateInfo trajectoryPoint = cachedTrajectory_.getStateAtHcal();
  // If trajectory point at HCAL is not valid, try to use the outer most state of the
  // trajectory instead.
  if (!trajectoryPoint.isValid())
    trajectoryPoint = cachedTrajectory_.getOuterState();
  if (!trajectoryPoint.isValid()) {
    LogTrace("TrackAssociator")
        << "trajectory position at HCAL is not valid. Assume the track cannot reach muon detectors and skip it";
    return;
  }

  GlobalVector direction = trajectoryPoint.momentum().unit();
  LogTrace("TrackAssociator") << "muon direction: " << direction
                              << "\n\t and corresponding point: " << trajectoryPoint.position() << "\n";

  DetIdAssociator::MapRange mapRange =
      getMapRange(cachedTrajectory_.trajectoryDelta(CachedTrajectory::FullTrajectory), parameters.dRMuonPreselection);

  // and find chamber DetIds

  std::set<DetId> muonIdsInRegion = muonDetIdAssociator_->getDetIdsCloseToAPoint(trajectoryPoint.position(), mapRange);
  LogTrace("TrackAssociator") << "Number of chambers to check: " << muonIdsInRegion.size();

  if (parameters.preselectMuonTracks) {
    std::set<DetId> muonIdsInRegionOccupied;
    std::set_intersection(muonIdsInRegion.begin(),
                          muonIdsInRegion.end(),
                          occupancy.begin(),
                          occupancy.end(),
                          std::inserter(muonIdsInRegionOccupied, muonIdsInRegionOccupied.begin()));
    if (muonIdsInRegionOccupied.empty())
      return;
  }

  for (std::set<DetId>::const_iterator detId = muonIdsInRegion.begin(); detId != muonIdsInRegion.end(); detId++) {
    const GeomDet* geomDet = muonDetIdAssociator_->getGeomDet(*detId);
    TrajectoryStateOnSurface stateOnSurface = cachedTrajectory_.propagate(&geomDet->surface());
    if (!stateOnSurface.isValid()) {
      LogTrace("TrackAssociator") << "Failed to propagate the track; moving on\n\t"
                                  << "Element is not crosssed: " << DetIdInfo::info(*detId, nullptr) << "\n";
      continue;
    }
    LocalPoint localPoint = geomDet->surface().toLocal(stateOnSurface.freeState()->position());
    LocalError localError = stateOnSurface.localError().positionError();
    float distanceX = 0.f;
    float distanceY = 0.f;
    if (const CSCChamber* cscChamber = dynamic_cast<const CSCChamber*>(geomDet)) {
      const CSCChamberSpecs* chamberSpecs = cscChamber->specs();
      if (!chamberSpecs) {
        LogTrace("TrackAssociator") << "Failed to get CSCChamberSpecs from CSCChamber; moving on\n";
        continue;
      }
      const CSCLayerGeometry* layerGeometry = chamberSpecs->oddLayerGeometry(1);
      if (!layerGeometry) {
        LogTrace("TrackAssociator") << "Failed to get CSCLayerGeometry from CSCChamberSpecs; moving on\n";
        continue;
      }
      const CSCWireTopology* wireTopology = layerGeometry->wireTopology();
      if (!wireTopology) {
        LogTrace("TrackAssociator") << "Failed to get CSCWireTopology from CSCLayerGeometry; moving on\n";
        continue;
      }

      float wideWidth = wireTopology->wideWidthOfPlane();
      float narrowWidth = wireTopology->narrowWidthOfPlane();
      float length = wireTopology->lengthOfPlane();
      // If slanted, there is no y offset between local origin and symmetry center of wire plane
      float yOfFirstWire = std::abs(wireTopology->wireAngle()) > 1.E-06f ? -0.5 * length : wireTopology->yOfWire(1);
      // y offset between local origin and symmetry center of wire plane
      float yCOWPOffset = yOfFirstWire + 0.5f * length;

      // tangent of the incline angle from inside the trapezoid
      float tangent = (wideWidth - narrowWidth) / (2.f * length);
      // y position wrt bottom of trapezoid
      float yPrime = localPoint.y() + std::abs(yOfFirstWire);
      // half trapezoid width at y' is 0.5 * narrowWidth + x side of triangle with the above tangent and side y'
      float halfWidthAtYPrime = 0.5f * narrowWidth + yPrime * tangent;
      distanceX = std::abs(localPoint.x()) - halfWidthAtYPrime;
      distanceY = std::abs(localPoint.y() - yCOWPOffset) - 0.5f * length;
    } else if (dynamic_cast<const GEMChamber*>(geomDet) || dynamic_cast<const GEMSuperChamber*>(geomDet)) {
      const TrapezoidalPlaneBounds* bounds = dynamic_cast<const TrapezoidalPlaneBounds*>(&geomDet->surface().bounds());

      float wideWidth = bounds->width();
      float narrowWidth = 2.f * bounds->widthAtHalfLength() - wideWidth;
      float length = bounds->length();
      float tangent = (wideWidth - narrowWidth) / (2.f * length);
      float halfWidthAtY = tangent * localPoint.y() + 0.25f * (narrowWidth + wideWidth);

      distanceX = std::abs(localPoint.x()) - halfWidthAtY;
      distanceY = std::abs(localPoint.y()) - 0.5f * length;
    } else {
      distanceX = std::abs(localPoint.x()) - 0.5f * geomDet->surface().bounds().width();
      distanceY = std::abs(localPoint.y()) - 0.5f * geomDet->surface().bounds().length();
    }
    if ((distanceX < parameters.muonMaxDistanceX && distanceY < parameters.muonMaxDistanceY) ||
        (distanceX * distanceX <
             localError.xx() * parameters.muonMaxDistanceSigmaX * parameters.muonMaxDistanceSigmaX &&
         distanceY * distanceY <
             localError.yy() * parameters.muonMaxDistanceSigmaY * parameters.muonMaxDistanceSigmaY)) {
      LogTrace("TrackAssociator") << "found a match: " << DetIdInfo::info(*detId, nullptr) << "\n";
      TAMuonChamberMatch match;
      match.tState = stateOnSurface;
      match.localDistanceX = distanceX;
      match.localDistanceY = distanceY;
      match.id = *detId;
      matches.push_back(match);
    } else {
      LogTrace("TrackAssociator") << "chamber is too far: " << DetIdInfo::info(*detId, nullptr)
                                  << "\n\tdistanceX: " << distanceX << "\t distanceY: " << distanceY
                                  << "\t sigmaX: " << distanceX / sqrt(localError.xx())
                                  << "\t sigmaY: " << distanceY / sqrt(localError.yy()) << "\n";
    }
  }
}

void TrackDetectorAssociator::fillMuon(const edm::Event& iEvent,
                                       TrackDetMatchInfo& info,
                                       const AssociatorParameters& parameters) {
  // Get the segments from the event
  edm::Handle<DTRecSegment4DCollection> dtSegments;
  iEvent.getByToken(parameters.dtSegmentsToken, dtSegments);
  if (!dtSegments.isValid())
    throw cms::Exception("FatalError") << "Unable to find DTRecSegment4DCollection in event!\n";

  edm::Handle<CSCSegmentCollection> cscSegments;
  iEvent.getByToken(parameters.cscSegmentsToken, cscSegments);
  if (!cscSegments.isValid())
    throw cms::Exception("FatalError") << "Unable to find CSCSegmentCollection in event!\n";

  edm::Handle<GEMSegmentCollection> gemSegments;
  if (parameters.useGEM)
    iEvent.getByToken(parameters.gemSegmentsToken, gemSegments);
  edm::Handle<ME0SegmentCollection> me0Segments;
  if (parameters.useME0)
    iEvent.getByToken(parameters.me0SegmentsToken, me0Segments);

  // Get the hits from the event only if track preselection is activated
  // Get the chambers segments/hits in the events
  std::set<DetId> occupancy_set;
  if (parameters.preselectMuonTracks) {
    edm::Handle<RPCRecHitCollection> rpcRecHits;
    iEvent.getByToken(parameters.rpcHitsToken, rpcRecHits);
    if (!rpcRecHits.isValid())
      throw cms::Exception("FatalError") << "Unable to find RPCRecHitCollection in event!\n";

    edm::Handle<GEMRecHitCollection> gemRecHits;
    if (parameters.useGEM)
      iEvent.getByToken(parameters.gemHitsToken, gemRecHits);

    edm::Handle<ME0RecHitCollection> me0RecHits;
    if (parameters.useME0)
      iEvent.getByToken(parameters.me0HitsToken, me0RecHits);

    for (const auto& dtSegment : *dtSegments) {
      occupancy_set.insert(dtSegment.geographicalId());
    }
    for (const auto& cscSegment : *cscSegments) {
      occupancy_set.insert(cscSegment.geographicalId());
    }
    for (const auto& rpcRecHit : *rpcRecHits) {
      occupancy_set.insert(rpcRecHit.geographicalId());
    }
    if (parameters.useGEM) {
      for (const auto& gemSegment : *gemSegments) {
        occupancy_set.insert(gemSegment.geographicalId());
      }
      for (const auto& gemRecHit : *gemRecHits) {
        occupancy_set.insert(gemRecHit.geographicalId());
      }
    }
    if (parameters.useME0) {
      for (const auto& me0Segment : *me0Segments) {
        occupancy_set.insert(me0Segment.geographicalId());
      }
      for (const auto& me0RecHit : *me0RecHits) {
        occupancy_set.insert(me0RecHit.geographicalId());
      }
    }
    if (occupancy_set.empty()) {
      LogTrace("TrackAssociator") << "No segments or hits were found in the event: aborting track extrapolation"
                                  << std::endl;
      return;
    }
  }

  ///// get a set of DetId's in a given direction

  // check the map of available segments
  // if there is no segments in a given direction at all,
  // then there is no point to fly there.
  //
  // MISSING
  // Possible solution: quick search for presence of segments
  // for the set of DetIds

  // get a set of matches corresponding to muon chambers
  std::vector<TAMuonChamberMatch> matchedChambers;
  LogTrace("TrackAssociator") << "Trying to Get ChamberMatches" << std::endl;
  getTAMuonChamberMatches(matchedChambers, parameters, occupancy_set);
  LogTrace("TrackAssociator") << "Chambers matched: " << matchedChambers.size() << "\n";

  // Iterate over all chamber matches and fill segment matching
  // info if it's available
  for (std::vector<TAMuonChamberMatch>::iterator matchedChamber = matchedChambers.begin();
       matchedChamber != matchedChambers.end();
       matchedChamber++) {
    const GeomDet* geomDet = muonDetIdAssociator_->getGeomDet((*matchedChamber).id);
    // DT chamber
    if (const DTChamber* chamber = dynamic_cast<const DTChamber*>(geomDet)) {
      // Get the range for the corresponding segments
      DTRecSegment4DCollection::range range = dtSegments->get(chamber->id());
      // Loop over the segments of this chamber
      for (DTRecSegment4DCollection::const_iterator segment = range.first; segment != range.second; segment++) {
        if (addTAMuonSegmentMatch(*matchedChamber, &(*segment), parameters)) {
          matchedChamber->segments.back().dtSegmentRef = DTRecSegment4DRef(dtSegments, segment - dtSegments->begin());
        }
      }
    }
    // CSC Chamber
    else if (const CSCChamber* chamber = dynamic_cast<const CSCChamber*>(geomDet)) {
      // Get the range for the corresponding segments
      CSCSegmentCollection::range range = cscSegments->get(chamber->id());
      // Loop over the segments
      for (CSCSegmentCollection::const_iterator segment = range.first; segment != range.second; segment++) {
        if (addTAMuonSegmentMatch(*matchedChamber, &(*segment), parameters)) {
          matchedChamber->segments.back().cscSegmentRef = CSCSegmentRef(cscSegments, segment - cscSegments->begin());
        }
      }
    } else {
      // GEM Chamber
      if (parameters.useGEM) {
        if (const GEMSuperChamber* chamber = dynamic_cast<const GEMSuperChamber*>(geomDet)) {
          // Get the range for the corresponding segments
          GEMSegmentCollection::range range = gemSegments->get(chamber->id());
          // Loop over the segments
          for (GEMSegmentCollection::const_iterator segment = range.first; segment != range.second; segment++) {
            if (addTAMuonSegmentMatch(*matchedChamber, &(*segment), parameters)) {
              matchedChamber->segments.back().gemSegmentRef =
                  GEMSegmentRef(gemSegments, segment - gemSegments->begin());
            }
          }
        }
      }
      // ME0 Chamber
      if (parameters.useME0) {
        if (const ME0Chamber* chamber = dynamic_cast<const ME0Chamber*>(geomDet)) {
          // Get the range for the corresponding segments
          ME0SegmentCollection::range range = me0Segments->get(chamber->id());
          // Loop over the segments
          for (ME0SegmentCollection::const_iterator segment = range.first; segment != range.second; segment++) {
            if (addTAMuonSegmentMatch(*matchedChamber, &(*segment), parameters)) {
              matchedChamber->segments.back().me0SegmentRef =
                  ME0SegmentRef(me0Segments, segment - me0Segments->begin());
            }
          }
        }
      }
    }
    info.chambers.push_back(*matchedChamber);
  }
}

bool TrackDetectorAssociator::addTAMuonSegmentMatch(TAMuonChamberMatch& matchedChamber,
                                                    const RecSegment* segment,
                                                    const AssociatorParameters& parameters) {
  LogTrace("TrackAssociator") << "Segment local position: " << segment->localPosition() << "\n"
                              << std::hex << segment->geographicalId().rawId() << "\n";

  const GeomDet* chamber = muonDetIdAssociator_->getGeomDet(matchedChamber.id);
  TrajectoryStateOnSurface trajectoryStateOnSurface = matchedChamber.tState;
  GlobalPoint segmentGlobalPosition = chamber->toGlobal(segment->localPosition());

  LogTrace("TrackAssociator") << "Segment global position: " << segmentGlobalPosition
                              << " \t (R_xy,eta,phi): " << segmentGlobalPosition.perp() << ","
                              << segmentGlobalPosition.eta() << "," << segmentGlobalPosition.phi() << "\n";

  LogTrace("TrackAssociator") << "\teta hit: " << segmentGlobalPosition.eta()
                              << " \tpropagator: " << trajectoryStateOnSurface.freeState()->position().eta() << "\n"
                              << "\tphi hit: " << segmentGlobalPosition.phi()
                              << " \tpropagator: " << trajectoryStateOnSurface.freeState()->position().phi()
                              << std::endl;

  bool isGood = false;
  bool isDTWithoutY = false;
  const DTRecSegment4D* dtseg = dynamic_cast<const DTRecSegment4D*>(segment);
  if (dtseg && (!dtseg->hasZed()))
    isDTWithoutY = true;

  float deltaPhi(std::abs(segmentGlobalPosition.phi() - trajectoryStateOnSurface.freeState()->position().phi()));
  if (deltaPhi > float(M_PI))
    deltaPhi = std::abs(deltaPhi - float(M_PI) * 2.f);
  float deltaEta = std::abs(segmentGlobalPosition.eta() - trajectoryStateOnSurface.freeState()->position().eta());

  if (isDTWithoutY) {
    isGood = deltaPhi < parameters.dRMuon;
    // Be in chamber
    isGood &= deltaEta < .3f;
  } else
    isGood = deltaEta * deltaEta + deltaPhi * deltaPhi < parameters.dRMuon * parameters.dRMuon;

  if (isGood) {
    TAMuonSegmentMatch muonSegment;
    muonSegment.segmentGlobalPosition = getPoint(segmentGlobalPosition);
    muonSegment.segmentLocalPosition = getPoint(segment->localPosition());
    muonSegment.segmentLocalDirection = getVector(segment->localDirection());
    muonSegment.segmentLocalErrorXX = segment->localPositionError().xx();
    muonSegment.segmentLocalErrorYY = segment->localPositionError().yy();
    muonSegment.segmentLocalErrorXY = segment->localPositionError().xy();
    muonSegment.segmentLocalErrorDxDz = segment->localDirectionError().xx();
    muonSegment.segmentLocalErrorDyDz = segment->localDirectionError().yy();

    // DANGEROUS - compiler cannot guaranty parameters ordering
    // AlgebraicSymMatrix segmentCovMatrix = segment->parametersError();
    // muonSegment.segmentLocalErrorXDxDz = segmentCovMatrix[2][0];
    // muonSegment.segmentLocalErrorYDyDz = segmentCovMatrix[3][1];
    muonSegment.segmentLocalErrorXDxDz = -999.f;
    muonSegment.segmentLocalErrorYDyDz = -999.f;
    muonSegment.hasZed = true;
    muonSegment.hasPhi = true;

    // timing information
    muonSegment.t0 = 0.f;
    if (dtseg) {
      if ((dtseg->hasPhi()) && (!isDTWithoutY)) {
        int phiHits = dtseg->phiSegment()->specificRecHits().size();
        //	  int zHits = dtseg->zSegment()->specificRecHits().size();
        int hits = 0;
        double t0 = 0.;
        // TODO: cuts on hit numbers not optimized in any way yet...
        if (phiHits > 5 && dtseg->phiSegment()->ist0Valid()) {
          t0 += dtseg->phiSegment()->t0() * phiHits;
          hits += phiHits;
          LogTrace("TrackAssociator") << " Phi t0: " << dtseg->phiSegment()->t0() << " hits: " << phiHits;
        }
        // the z segments seem to contain little useful information...
        //	  if (zHits>3) {
        //	    t0+=s->zSegment()->t0()*zHits;
        //	    hits+=zHits;
        //	    LogTrace("TrackAssociator") << "   Z t0: " << s->zSegment()->t0() << " hits: " << zHits << std::endl;
        //	  }
        if (hits)
          muonSegment.t0 = t0 / hits;
        //	  LogTrace("TrackAssociator") << " --- t0: " << muonSegment.t0 << std::endl;
      } else {
        // check and set dimensionality
        if (isDTWithoutY)
          muonSegment.hasZed = false;
        if (!dtseg->hasPhi())
          muonSegment.hasPhi = false;
      }
    }
    matchedChamber.segments.push_back(muonSegment);
  }

  return isGood;
}

//********************** NON-CORE CODE ******************************//

void TrackDetectorAssociator::fillCaloTruth(const edm::Event& iEvent,
                                            TrackDetMatchInfo& info,
                                            const AssociatorParameters& parameters) {
  // get list of simulated tracks and their vertices
  using namespace edm;
  Handle<SimTrackContainer> simTracks;
  iEvent.getByToken(parameters.simTracksToken, simTracks);
  if (!simTracks.isValid())
    throw cms::Exception("FatalError") << "No simulated tracks found\n";

  Handle<SimVertexContainer> simVertices;
  iEvent.getByToken(parameters.simVerticesToken, simVertices);
  if (!simVertices.isValid())
    throw cms::Exception("FatalError") << "No simulated vertices found\n";

  // get sim calo hits
  Handle<PCaloHitContainer> simEcalHitsEB;
  iEvent.getByToken(parameters.simEcalHitsEBToken, simEcalHitsEB);
  if (!simEcalHitsEB.isValid())
    throw cms::Exception("FatalError") << "No simulated ECAL EB hits found\n";

  Handle<PCaloHitContainer> simEcalHitsEE;
  iEvent.getByToken(parameters.simEcalHitsEEToken, simEcalHitsEE);
  if (!simEcalHitsEE.isValid())
    throw cms::Exception("FatalError") << "No simulated ECAL EE hits found\n";

  Handle<PCaloHitContainer> simHcalHits;
  iEvent.getByToken(parameters.simHcalHitsToken, simHcalHits);
  if (!simHcalHits.isValid())
    throw cms::Exception("FatalError") << "No simulated HCAL hits found\n";

  // find truth partner
  SimTrackContainer::const_iterator simTrack = simTracks->begin();
  for (; simTrack != simTracks->end(); ++simTrack) {
    math::XYZVector simP3(simTrack->momentum().x(), simTrack->momentum().y(), simTrack->momentum().z());
    math::XYZVector recoP3(info.stateAtIP.momentum().x(), info.stateAtIP.momentum().y(), info.stateAtIP.momentum().z());
    if (ROOT::Math::VectorUtil::DeltaR(recoP3, simP3) < 0.1)
      break;
  }
  if (simTrack != simTracks->end()) {
    info.simTrack = &(*simTrack);
    float ecalTrueEnergy(0);
    float hcalTrueEnergy(0);

    // loop over calo hits
    for (PCaloHitContainer::const_iterator hit = simEcalHitsEB->begin(); hit != simEcalHitsEB->end(); ++hit)
      if (hit->geantTrackId() == info.simTrack->genpartIndex())
        ecalTrueEnergy += hit->energy();

    for (PCaloHitContainer::const_iterator hit = simEcalHitsEE->begin(); hit != simEcalHitsEE->end(); ++hit)
      if (hit->geantTrackId() == info.simTrack->genpartIndex())
        ecalTrueEnergy += hit->energy();

    for (PCaloHitContainer::const_iterator hit = simHcalHits->begin(); hit != simHcalHits->end(); ++hit)
      if (hit->geantTrackId() == info.simTrack->genpartIndex())
        hcalTrueEnergy += hit->energy();

    info.ecalTrueEnergy = ecalTrueEnergy;
    info.hcalTrueEnergy = hcalTrueEnergy;
    info.hcalTrueEnergyCorrected = hcalTrueEnergy;
    if (std::abs(info.trkGlobPosAtHcal.eta()) < 1.3f)
      info.hcalTrueEnergyCorrected = hcalTrueEnergy * 113.2f;
    else if (std::abs(info.trkGlobPosAtHcal.eta()) < 3.0f)
      info.hcalTrueEnergyCorrected = hcalTrueEnergy * 167.2f;
  }
}

TrackDetMatchInfo TrackDetectorAssociator::associate(const edm::Event& iEvent,
                                                     const edm::EventSetup& iSetup,
                                                     const reco::Track& track,
                                                     const AssociatorParameters& parameters,
                                                     Direction direction /*= Any*/) {
  double currentStepSize = cachedTrajectory_.getPropagationStep();

  const MagneticField* bField = &iSetup.getData(parameters.bFieldToken);

  if (track.extra().isNull()) {
    if (direction != InsideOut)
      throw cms::Exception("FatalError") << "No TrackExtra information is available and association is done with "
                                            "something else than InsideOut track.\n"
                                         << "Either change the parameter or provide needed data!\n";
    LogTrace("TrackAssociator") << "Track Extras not found\n";
    FreeTrajectoryState initialState = trajectoryStateTransform::initialFreeState(track, bField);
    return associate(iEvent, iSetup, parameters, &initialState);  // 5th argument is null pointer
  }

  LogTrace("TrackAssociator") << "Track Extras found\n";
  FreeTrajectoryState innerState = trajectoryStateTransform::innerFreeState(track, bField);
  FreeTrajectoryState outerState = trajectoryStateTransform::outerFreeState(track, bField);
  FreeTrajectoryState referenceState = trajectoryStateTransform::initialFreeState(track, bField);

  LogTrace("TrackAssociator") << "inner track state (rho, z, phi):" << track.innerPosition().Rho() << ", "
                              << track.innerPosition().z() << ", " << track.innerPosition().phi() << "\n";
  LogTrace("TrackAssociator") << "innerFreeState (rho, z, phi):" << innerState.position().perp() << ", "
                              << innerState.position().z() << ", " << innerState.position().phi() << "\n";

  LogTrace("TrackAssociator") << "outer track state (rho, z, phi):" << track.outerPosition().Rho() << ", "
                              << track.outerPosition().z() << ", " << track.outerPosition().phi() << "\n";
  LogTrace("TrackAssociator") << "outerFreeState (rho, z, phi):" << outerState.position().perp() << ", "
                              << outerState.position().z() << ", " << outerState.position().phi() << "\n";

  // InsideOut first
  if (crossedIP(track)) {
    switch (direction) {
      case InsideOut:
      case Any:
        return associate(iEvent, iSetup, parameters, &referenceState, &outerState);
        break;
      case OutsideIn: {
        cachedTrajectory_.setPropagationStep(-std::abs(currentStepSize));
        TrackDetMatchInfo result = associate(iEvent, iSetup, parameters, &innerState, &referenceState);
        cachedTrajectory_.setPropagationStep(currentStepSize);
        return result;
        break;
      }
    }
  } else {
    switch (direction) {
      case InsideOut:
        return associate(iEvent, iSetup, parameters, &innerState, &outerState);
        break;
      case OutsideIn: {
        cachedTrajectory_.setPropagationStep(-std::abs(currentStepSize));
        TrackDetMatchInfo result = associate(iEvent, iSetup, parameters, &outerState, &innerState);
        cachedTrajectory_.setPropagationStep(currentStepSize);
        return result;
        break;
      }
      case Any: {
        // check if we deal with clear outside-in case
        if (track.innerPosition().Dot(track.innerMomentum()) < 0 &&
            track.outerPosition().Dot(track.outerMomentum()) < 0) {
          cachedTrajectory_.setPropagationStep(-std::abs(currentStepSize));
          TrackDetMatchInfo result;
          if (track.innerPosition().Mag2() < track.outerPosition().Mag2())
            result = associate(iEvent, iSetup, parameters, &innerState, &outerState);
          else
            result = associate(iEvent, iSetup, parameters, &outerState, &innerState);
          cachedTrajectory_.setPropagationStep(currentStepSize);
          return result;
        }
      }
    }
  }

  // all other cases
  return associate(iEvent, iSetup, parameters, &innerState, &outerState);
}

TrackDetMatchInfo TrackDetectorAssociator::associate(const edm::Event& iEvent,
                                                     const edm::EventSetup& iSetup,
                                                     const SimTrack& track,
                                                     const SimVertex& vertex,
                                                     const AssociatorParameters& parameters) {
  auto const* bField = &iSetup.getData(parameters.bFieldToken);
  return associate(iEvent, iSetup, getFreeTrajectoryState(bField, track, vertex), parameters);
}

TrackDetMatchInfo TrackDetectorAssociator::associate(const edm::Event& iEvent,
                                                     const edm::EventSetup& iSetup,
                                                     const GlobalVector& momentum,
                                                     const GlobalPoint& vertex,
                                                     const int charge,
                                                     const AssociatorParameters& parameters) {
  auto const* bField = &iSetup.getData(parameters.bFieldToken);
  return associate(iEvent, iSetup, getFreeTrajectoryState(bField, momentum, vertex, charge), parameters);
}

bool TrackDetectorAssociator::crossedIP(const reco::Track& track) {
  bool crossed = true;
  crossed &= (track.innerPosition().rho() > 3);  // something close to active volume
  crossed &= (track.outerPosition().rho() > 3);  // something close to active volume
  crossed &=
      ((track.innerPosition().x() * track.innerMomentum().x() + track.innerPosition().y() * track.innerMomentum().y() <
        0) !=
       (track.outerPosition().x() * track.outerMomentum().x() + track.outerPosition().y() * track.outerMomentum().y() <
        0));
  return crossed;
}
