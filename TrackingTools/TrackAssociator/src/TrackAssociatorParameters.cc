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

#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"


void TrackAssociatorParameters::loadParameters( const edm::ParameterSet& iConfig, edm::ConsumesCollector &iC )
{
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
   useHO   = iConfig.getParameter<bool>("useHO");
   useCalo = iConfig.getParameter<bool>("useCalo");
   useMuon = iConfig.getParameter<bool>("useMuon");
   usePreshower = iConfig.getParameter<bool>("usePreshower");
   
   theEBRecHitCollectionLabel       = iConfig.getParameter<edm::InputTag>("EBRecHitCollectionLabel");
   theEERecHitCollectionLabel       = iConfig.getParameter<edm::InputTag>("EERecHitCollectionLabel");
   theCaloTowerCollectionLabel      = iConfig.getParameter<edm::InputTag>("CaloTowerCollectionLabel");
   theHBHERecHitCollectionLabel     = iConfig.getParameter<edm::InputTag>("HBHERecHitCollectionLabel");
   theHORecHitCollectionLabel       = iConfig.getParameter<edm::InputTag>("HORecHitCollectionLabel");
   theDTRecSegment4DCollectionLabel = iConfig.getParameter<edm::InputTag>("DTRecSegment4DCollectionLabel");
   theCSCSegmentCollectionLabel     = iConfig.getParameter<edm::InputTag>("CSCSegmentCollectionLabel");
   theGEMSegmentCollectionLabel     = iConfig.getParameter<edm::InputTag>("GEMSegmentCollectionLabel");

   accountForTrajectoryChangeCalo   = iConfig.getParameter<bool>("accountForTrajectoryChangeCalo");
   // accountForTrajectoryChangeMuon   = iConfig.getParameter<bool>("accountForTrajectoryChangeMuon");
   
   truthMatch = iConfig.getParameter<bool>("truthMatch");
   muonMaxDistanceSigmaY = iConfig.getParameter<double>("trajectoryUncertaintyTolerance");

   if (useEcal) {
     EBRecHitsToken=iC.consumes<EBRecHitCollection>(theEBRecHitCollectionLabel);
     EERecHitsToken=iC.consumes<EERecHitCollection>(theEERecHitCollectionLabel);
   }
   if (useCalo) caloTowersToken=iC.consumes<CaloTowerCollection>(theCaloTowerCollectionLabel);
   if (useHcal) HBHEcollToken=iC.consumes<HBHERecHitCollection>(theHBHERecHitCollectionLabel);
   if (useHO) HOcollToken=iC.consumes<HORecHitCollection>(theHORecHitCollectionLabel);
   if (useMuon) {
     dtSegmentsToken=iC.consumes<DTRecSegment4DCollection>(theDTRecSegment4DCollectionLabel);
     cscSegmentsToken=iC.consumes<CSCSegmentCollection>(theCSCSegmentCollectionLabel);
     gemSegmentsToken=iC.consumes<GEMSegmentCollection>(theGEMSegmentCollectionLabel);
   }
   if (truthMatch) {
     simTracksToken=iC.consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
     simVerticesToken=iC.consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
     simEcalHitsEBToken=iC.consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","EcalHitsEB"));
     simEcalHitsEEToken=iC.consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","EcalHitsEE"));
     simHcalHitsToken=iC.consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits","HcalHits"));
   }
}

TrackAssociatorParameters::TrackAssociatorParameters( const edm::ParameterSet& iConfig, edm::ConsumesCollector &&iC )
{
  loadParameters( iConfig, iC );
}

