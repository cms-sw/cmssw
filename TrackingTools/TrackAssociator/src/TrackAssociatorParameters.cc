// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      TrackAssociatorParameters
//
/*

 Description: track associator parameters

*/
//
// Original Author:  Dmytro Kovalskyi
//
//
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

void TrackAssociatorParameters::loadParameters(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC) {
  dREcal = iConfig.getParameter<double>("dREcal");
  dRHcal = iConfig.getParameter<double>("dRHcal");
  dRMuon = iConfig.getParameter<double>("dRMuon");

  dREcalPreselection = iConfig.getParameter<double>("dREcalPreselection");
  dRHcalPreselection = iConfig.getParameter<double>("dRHcalPreselection");
  dRMuonPreselection = iConfig.getParameter<double>("dRMuonPreselection");
  dRPreshowerPreselection = iConfig.getParameter<double>("dRPreshowerPreselection");

  muonMaxDistanceX = iConfig.getParameter<double>("muonMaxDistanceX");
  muonMaxDistanceY = iConfig.getParameter<double>("muonMaxDistanceY");
  muonMaxDistanceSigmaX = iConfig.getParameter<double>("muonMaxDistanceSigmaX");
  muonMaxDistanceSigmaY = iConfig.getParameter<double>("muonMaxDistanceSigmaY");

  useEcal = iConfig.getParameter<bool>("useEcal");
  useHcal = iConfig.getParameter<bool>("useHcal");
  useHO = iConfig.getParameter<bool>("useHO");
  useCalo = iConfig.getParameter<bool>("useCalo");
  useMuon = iConfig.getParameter<bool>("useMuon");
  usePreshower = iConfig.getParameter<bool>("usePreshower");
  useGEM = iConfig.getParameter<bool>("useGEM");
  useME0 = iConfig.getParameter<bool>("useME0");
  preselectMuonTracks = iConfig.getParameter<bool>("preselectMuonTracks");

  theEBRecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("EBRecHitCollectionLabel");
  theEERecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("EERecHitCollectionLabel");
  theCaloTowerCollectionLabel = iConfig.getParameter<edm::InputTag>("CaloTowerCollectionLabel");
  theHBHERecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("HBHERecHitCollectionLabel");
  theHORecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("HORecHitCollectionLabel");
  theDTRecSegment4DCollectionLabel = iConfig.getParameter<edm::InputTag>("DTRecSegment4DCollectionLabel");
  theCSCSegmentCollectionLabel = iConfig.getParameter<edm::InputTag>("CSCSegmentCollectionLabel");
  theGEMSegmentCollectionLabel = iConfig.getParameter<edm::InputTag>("GEMSegmentCollectionLabel");
  theME0SegmentCollectionLabel = iConfig.getParameter<edm::InputTag>("ME0SegmentCollectionLabel");
  if (preselectMuonTracks) {
    theRPCHitCollectionLabel = iConfig.getParameter<edm::InputTag>("RPCHitCollectionLabel");
    theGEMHitCollectionLabel = iConfig.getParameter<edm::InputTag>("GEMHitCollectionLabel");
    theME0HitCollectionLabel = iConfig.getParameter<edm::InputTag>("ME0HitCollectionLabel");
  }

  accountForTrajectoryChangeCalo = iConfig.getParameter<bool>("accountForTrajectoryChangeCalo");
  // accountForTrajectoryChangeMuon   = iConfig.getParameter<bool>("accountForTrajectoryChangeMuon");

  truthMatch = iConfig.getParameter<bool>("truthMatch");
  muonMaxDistanceSigmaY = iConfig.getParameter<double>("trajectoryUncertaintyTolerance");

  if (useEcal) {
    EBRecHitsToken = iC.consumes<EBRecHitCollection>(theEBRecHitCollectionLabel);
    EERecHitsToken = iC.consumes<EERecHitCollection>(theEERecHitCollectionLabel);
  }
  if (useCalo)
    caloTowersToken = iC.consumes<CaloTowerCollection>(theCaloTowerCollectionLabel);
  if (useHcal)
    HBHEcollToken = iC.consumes<HBHERecHitCollection>(theHBHERecHitCollectionLabel);
  if (useHO)
    HOcollToken = iC.consumes<HORecHitCollection>(theHORecHitCollectionLabel);
  if (useMuon) {
    dtSegmentsToken = iC.consumes<DTRecSegment4DCollection>(theDTRecSegment4DCollectionLabel);
    cscSegmentsToken = iC.consumes<CSCSegmentCollection>(theCSCSegmentCollectionLabel);
    if (useGEM)
      gemSegmentsToken = iC.consumes<GEMSegmentCollection>(theGEMSegmentCollectionLabel);
    if (useME0)
      me0SegmentsToken = iC.consumes<ME0SegmentCollection>(theME0SegmentCollectionLabel);
    if (preselectMuonTracks) {
      rpcHitsToken = iC.consumes<RPCRecHitCollection>(theRPCHitCollectionLabel);
      gemHitsToken = iC.consumes<GEMRecHitCollection>(theGEMHitCollectionLabel);
      me0HitsToken = iC.consumes<ME0RecHitCollection>(theME0HitCollectionLabel);
    }
  }
  if (truthMatch) {
    simTracksToken = iC.consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
    simVerticesToken = iC.consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
    simEcalHitsEBToken = iC.consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEB"));
    simEcalHitsEEToken = iC.consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEE"));
    simHcalHitsToken = iC.consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "HcalHits"));
  }

  ecalDetIdAssociatorToken = iC.esConsumes(edm::ESInputTag("", "EcalDetIdAssociator"));
  hcalDetIdAssociatorToken = iC.esConsumes(edm::ESInputTag("", "HcalDetIdAssociator"));
  hoDetIdAssociatorToken = iC.esConsumes(edm::ESInputTag("", "HODetIdAssociator"));
  caloDetIdAssociatorToken = iC.esConsumes(edm::ESInputTag("", "CaloDetIdAssociator"));
  muonDetIdAssociatorToken = iC.esConsumes(edm::ESInputTag("", "MuonDetIdAssociator"));
  preshowerDetIdAssociatorToken = iC.esConsumes(edm::ESInputTag("", "PreshowerDetIdAssociator"));
  theCaloGeometryToken = iC.esConsumes();
  theTrackingGeometryToken = iC.esConsumes();
  bFieldToken = iC.esConsumes();
}

TrackAssociatorParameters::TrackAssociatorParameters(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC) {
  loadParameters(iConfig, iC);
}

void TrackAssociatorParameters::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.setComment("Auxilliary class to store parameters for track association");
  // the following setup is the one from TrackingTools/TrackAssociator/python/default_cfi.py
  desc.add<bool>("accountForTrajectoryChangeCalo", false);
  desc.add<bool>("propagateAllDirections", true);
  desc.add<bool>("truthMatch", false);
  desc.add<bool>("useCalo", false);
  desc.add<bool>("useEcal", true);
  desc.add<bool>("useGEM", false);
  desc.add<bool>("useHO", true);
  desc.add<bool>("useHcal", true);
  desc.add<bool>("useME0", false);
  desc.add<bool>("useMuon", true);
  desc.add<bool>("usePreshower", false);
  desc.add<bool>("preselectMuonTracks", false);
  desc.add<double>("dREcal", 9999.0);
  desc.add<double>("dREcalPreselection", 0.05);
  desc.add<double>("dRHcal", 9999.0);
  desc.add<double>("dRHcalPreselection", 0.2);
  desc.add<double>("dRMuon", 9999.0);
  desc.add<double>("dRMuonPreselection", 0.2);
  desc.add<double>("dRPreshowerPreselection", 0.2);
  desc.add<double>("muonMaxDistanceSigmaX", 0.0);
  desc.add<double>("muonMaxDistanceSigmaY", 0.0);
  desc.add<double>("muonMaxDistanceX", 5.0);
  desc.add<double>("muonMaxDistanceY", 5.0);
  desc.add<double>("trajectoryUncertaintyTolerance", -1.0);
  desc.add<edm::InputTag>("CSCSegmentCollectionLabel", edm::InputTag("cscSegments"));
  desc.add<edm::InputTag>("CaloTowerCollectionLabel", edm::InputTag("towerMaker"));
  desc.add<edm::InputTag>("DTRecSegment4DCollectionLabel", edm::InputTag("dt4DSegments"));
  desc.add<edm::InputTag>("EBRecHitCollectionLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("EERecHitCollectionLabel", edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  desc.add<edm::InputTag>("GEMSegmentCollectionLabel", edm::InputTag("gemSegments"));
  desc.add<edm::InputTag>("HBHERecHitCollectionLabel", edm::InputTag("hbreco"));
  desc.add<edm::InputTag>("HORecHitCollectionLabel", edm::InputTag("horeco"));
  desc.add<edm::InputTag>("ME0SegmentCollectionLabel", edm::InputTag("me0Segments"));
  desc.add<edm::InputTag>("RPCHitCollectionLabel", edm::InputTag("rpcRecHits"));
  desc.add<edm::InputTag>("GEMHitCollectionLabel", edm::InputTag("gemRecHits"));
  desc.add<edm::InputTag>("ME0HitCollectionLabel", edm::InputTag("me0RecHits"));
}
