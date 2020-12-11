// -*- C++ -*-
//
// Package:     SiPixelPhase1HitsV
// Class:       SiPixelPhase1HitsV
//

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Validation/SiPixelPhase1HitsV/interface/SiPixelPhase1HitsV.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// class TrackAssociatorByHits;

SiPixelPhase1HitsV::SiPixelPhase1HitsV(const edm::ParameterSet &iConfig)
    : SiPixelPhase1Base(iConfig),
      pixelBarrelLowToken_(consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("pixBarrelLowSrc"))),
      pixelBarrelHighToken_(consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("pixBarrelHighSrc"))),
      pixelForwardLowToken_(consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("pixForwardLowSrc"))),
      pixelForwardHighToken_(consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("pixForwardHighSrc"))),

      tracksToken_(consumes<edm::View<reco::Track>>(iConfig.getParameter<edm::InputTag>("tracksTag"))),
      tpToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("tpTag"))),
      trackAssociatorByHitsToken_(consumes<reco::TrackToTrackingParticleAssociator>(
          iConfig.getParameter<edm::InputTag>("trackAssociatorByHitsTag"))),

      trackerGeomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()) {}

void SiPixelPhase1HitsV::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<edm::PSimHitContainer> barrelLowInput;
  iEvent.getByToken(pixelBarrelLowToken_, barrelLowInput);
  if (!barrelLowInput.isValid())
    return;

  edm::Handle<edm::PSimHitContainer> barrelHighInput;
  iEvent.getByToken(pixelBarrelHighToken_, barrelHighInput);
  if (!barrelHighInput.isValid())
    return;

  edm::Handle<edm::PSimHitContainer> forwardLowInput;
  iEvent.getByToken(pixelForwardLowToken_, forwardLowInput);
  if (!forwardLowInput.isValid())
    return;

  edm::Handle<edm::PSimHitContainer> forwardHighInput;
  iEvent.getByToken(pixelForwardHighToken_, forwardHighInput);
  if (!forwardHighInput.isValid())
    return;

  edm::PSimHitContainer::const_iterator it;

  // Get geometry information

  edm::ESHandle<TrackerGeometry> tracker = iSetup.getHandle(trackerGeomToken_);

  // get low barrel info
  for (it = barrelLowInput->begin(); it != barrelLowInput->end(); ++it) {
    auto id = DetId(it->detUnitId());
    const GeomDetUnit *det = (const GeomDetUnit *)tracker->idToDetUnit(id);
    GlobalPoint gpos = det->toGlobal(it->localPosition());

    float tof = it->timeOfFlight();
    float globalR = gpos.mag();

    float energyLoss = it->energyLoss();

    float entryExitX = (it->entryPoint().x() - it->exitPoint().x());
    float entryExitY = (it->entryPoint().y() - it->exitPoint().y());
    float entryExitZ = std::abs(it->entryPoint().z() - it->exitPoint().z());

    float localX = it->localPosition().x();
    float localY = it->localPosition().y();
    float localZ = it->localPosition().z();
    float localPhi = it->localPosition().phi();
    float localEta = it->localPosition().eta();

    histo[TOF_R].fill(globalR, tof, id, &iEvent);
    histo[ELOSS].fill(energyLoss, id, &iEvent);
    histo[ENTRY_EXIT_X].fill(entryExitX, id, &iEvent);
    histo[ENTRY_EXIT_Y].fill(entryExitY, id, &iEvent);
    histo[ENTRY_EXIT_Z].fill(entryExitZ, id, &iEvent);
    histo[LOCAL_X].fill(localX, id, &iEvent);
    histo[LOCAL_Y].fill(localY, id, &iEvent);
    histo[LOCAL_Z].fill(localZ, id, &iEvent);
    histo[LOCAL_PHI].fill(localPhi, id, &iEvent);
    histo[LOCAL_ETA].fill(localEta, id, &iEvent);
  }
  // get high barrel info
  for (it = barrelHighInput->begin(); it != barrelHighInput->end(); ++it) {
    auto id = DetId(it->detUnitId());
    const GeomDetUnit *det = (const GeomDetUnit *)tracker->idToDetUnit(id);
    GlobalPoint gpos = det->toGlobal(it->localPosition());

    float tof = it->timeOfFlight();
    float globalR = gpos.mag();

    float energyLoss = it->energyLoss();

    float entryExitX = (it->entryPoint().x() - it->exitPoint().x());
    float entryExitY = (it->entryPoint().y() - it->exitPoint().y());
    float entryExitZ = std::abs(it->entryPoint().z() - it->exitPoint().z());

    float localX = it->localPosition().x();
    float localY = it->localPosition().y();
    float localZ = it->localPosition().z();
    float localPhi = it->localPosition().phi();
    float localEta = it->localPosition().eta();

    histo[TOF_R].fill(globalR, tof, id, &iEvent);
    histo[ELOSS].fill(energyLoss, id, &iEvent);
    histo[ENTRY_EXIT_X].fill(entryExitX, id, &iEvent);
    histo[ENTRY_EXIT_Y].fill(entryExitY, id, &iEvent);
    histo[ENTRY_EXIT_Z].fill(entryExitZ, id, &iEvent);
    histo[LOCAL_X].fill(localX, id, &iEvent);
    histo[LOCAL_Y].fill(localY, id, &iEvent);
    histo[LOCAL_Z].fill(localZ, id, &iEvent);
    histo[LOCAL_PHI].fill(localPhi, id, &iEvent);
    histo[LOCAL_ETA].fill(localEta, id, &iEvent);
  }

  // get low forward info
  for (it = forwardLowInput->begin(); it != forwardLowInput->end(); ++it) {
    auto id = DetId(it->detUnitId());
    const GeomDetUnit *det = (const GeomDetUnit *)tracker->idToDetUnit(id);
    GlobalPoint gpos = det->toGlobal(it->localPosition());

    float tof = it->timeOfFlight();
    float globalR = gpos.mag();

    float energyLoss = it->energyLoss();

    float entryExitX = (it->entryPoint().x() - it->exitPoint().x());
    float entryExitY = (it->entryPoint().y() - it->exitPoint().y());
    float entryExitZ = std::abs(it->entryPoint().z() - it->exitPoint().z());

    float localX = it->localPosition().x();
    float localY = it->localPosition().y();
    float localZ = it->localPosition().z();
    float localPhi = it->localPosition().phi();
    float localEta = it->localPosition().eta();

    histo[TOF_R].fill(globalR, tof, id, &iEvent);
    histo[ELOSS].fill(energyLoss, id, &iEvent);
    histo[ENTRY_EXIT_X].fill(entryExitX, id, &iEvent);
    histo[ENTRY_EXIT_Y].fill(entryExitY, id, &iEvent);
    histo[ENTRY_EXIT_Z].fill(entryExitZ, id, &iEvent);
    histo[LOCAL_X].fill(localX, id, &iEvent);
    histo[LOCAL_Y].fill(localY, id, &iEvent);
    histo[LOCAL_Z].fill(localZ, id, &iEvent);
    histo[LOCAL_PHI].fill(localPhi, id, &iEvent);
    histo[LOCAL_ETA].fill(localEta, id, &iEvent);
  }

  // get high forward info
  for (it = forwardHighInput->begin(); it != forwardHighInput->end(); ++it) {
    auto id = DetId(it->detUnitId());
    const GeomDetUnit *det = (const GeomDetUnit *)tracker->idToDetUnit(id);
    GlobalPoint gpos = det->toGlobal(it->localPosition());

    float tof = it->timeOfFlight();
    float globalR = gpos.mag();

    float energyLoss = it->energyLoss();

    float entryExitX = (it->entryPoint().x() - it->exitPoint().x());
    float entryExitY = (it->entryPoint().y() - it->exitPoint().y());
    float entryExitZ = std::abs(it->entryPoint().z() - it->exitPoint().z());

    float localX = it->localPosition().x();
    float localY = it->localPosition().y();
    float localZ = it->localPosition().z();
    float localPhi = it->localPosition().phi();
    float localEta = it->localPosition().eta();

    histo[TOF_R].fill(globalR, tof, id, &iEvent);
    histo[ELOSS].fill(energyLoss, id, &iEvent);
    histo[ENTRY_EXIT_X].fill(entryExitX, id, &iEvent);
    histo[ENTRY_EXIT_Y].fill(entryExitY, id, &iEvent);
    histo[ENTRY_EXIT_Z].fill(entryExitZ, id, &iEvent);
    histo[LOCAL_X].fill(localX, id, &iEvent);
    histo[LOCAL_Y].fill(localY, id, &iEvent);
    histo[LOCAL_Z].fill(localZ, id, &iEvent);
    histo[LOCAL_PHI].fill(localPhi, id, &iEvent);
    histo[LOCAL_ETA].fill(localEta, id, &iEvent);
  }

  // Sim Hit efficiency info
  edm::Handle<edm::View<reco::Track>> trackCollectionH;
  iEvent.getByToken(tracksToken_, trackCollectionH);
  const edm::View<reco::Track> &tC = *(trackCollectionH.product());

  edm::Handle<TrackingParticleCollection> TPCollectionH;
  iEvent.getByToken(tpToken_, TPCollectionH);

  edm::Handle<reco::TrackToTrackingParticleAssociator> theHitsAssociator;
  iEvent.getByToken(trackAssociatorByHitsToken_, theHitsAssociator);
  if (!theHitsAssociator.isValid()) {
    throw cms::Exception("NO VALID HIT ASSOCIATOR");
  }
  reco::TrackToTrackingParticleAssociator const *associatorByHits = theHitsAssociator.product();

  if (TPCollectionH.isValid() && trackCollectionH.isValid()) {
    reco::RecoToSimCollection const &p = associatorByHits->associateRecoToSim(trackCollectionH, TPCollectionH);

    for (edm::View<reco::Track>::size_type i = 0; i < tC.size(); ++i) {
      edm::RefToBase<reco::Track> track(trackCollectionH, i);
      //      const reco::Track& t = *track;
      auto id = DetId(track->innerDetId());  // histo manager requires a det ID,
                                             // use innermost ID for ease

      auto iter = p.find(track);
      histo[EFFICIENCY_TRACK].fill(iter != p.end() ? 1 : 0, id, &iEvent);
    }
  }
}

DEFINE_FWK_MODULE(SiPixelPhase1HitsV);
