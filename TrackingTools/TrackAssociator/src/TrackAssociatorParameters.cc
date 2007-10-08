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
// $Id: TrackAssociatorParameters.cc,v 1.3.4.1 2007/10/08 10:28:18 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"


void TrackAssociatorParameters::loadParameters( const edm::ParameterSet& iConfig )
{
   dREcal = iConfig.getParameter<double>("dREcal");
   dRHcal = iConfig.getParameter<double>("dRHcal");
   dRMuon = iConfig.getParameter<double>("dRMuon");
   
   dREcalPreselection = iConfig.getParameter<double>("dREcalPreselection");
   dRHcalPreselection = iConfig.getParameter<double>("dRHcalPreselection");
   dRMuonPreselection = iConfig.getParameter<double>("dRMuonPreselection");
   
   muonMaxDistanceX = iConfig.getParameter<double>("muonMaxDistanceX");
   muonMaxDistanceY = iConfig.getParameter<double>("muonMaxDistanceY");
   muonMaxDistanceSigmaX = iConfig.getParameter<double>("muonMaxDistanceSigmaX");
   muonMaxDistanceSigmaY = iConfig.getParameter<double>("muonMaxDistanceSigmaY");
   
   useEcal = iConfig.getParameter<bool>("useEcal");
   useHcal = iConfig.getParameter<bool>("useHcal");
   useHO   = iConfig.getParameter<bool>("useHO");
   useCalo = iConfig.getParameter<bool>("useCalo");
   useMuon = iConfig.getParameter<bool>("useMuon");
   
   theEBRecHitCollectionLabel       = iConfig.getParameter<edm::InputTag>("EBRecHitCollectionLabel");
   theEERecHitCollectionLabel       = iConfig.getParameter<edm::InputTag>("EERecHitCollectionLabel");
   theCaloTowerCollectionLabel      = iConfig.getParameter<edm::InputTag>("CaloTowerCollectionLabel");
   theHBHERecHitCollectionLabel     = iConfig.getParameter<edm::InputTag>("HBHERecHitCollectionLabel");
   theHORecHitCollectionLabel       = iConfig.getParameter<edm::InputTag>("HORecHitCollectionLabel");
   theDTRecSegment4DCollectionLabel = iConfig.getParameter<edm::InputTag>("DTRecSegment4DCollectionLabel");
   theCSCSegmentCollectionLabel     = iConfig.getParameter<edm::InputTag>("CSCSegmentCollectionLabel");
   
   accountForTrajectoryChangeCalo   = iConfig.getParameter<bool>("accountForTrajectoryChangeCalo");
   // accountForTrajectoryChangeMuon   = iConfig.getParameter<bool>("accountForTrajectoryChangeMuon");
   
   truthMatch = iConfig.getParameter<bool>("truthMatch");
   std::string type = iConfig.getParameter<std::string>("crossedEnergyType");
   
   crossedEnergyType = SinglePointAlongTrajectory;    // default
   if ( type == "FivePointTwoSigmaElipseAlongTrajectory" )  
     crossedEnergyType = FivePointTwoSigmaElipseAlongTrajectory;
   else
     if ( type != "SinglePointAlongTrajectory" ) 
       throw cms::Exception("ConfigurationError") << "Uknown algorithm type: " << type << "\n";
}

TrackAssociatorParameters::TrackAssociatorParameters( const edm::ParameterSet& iConfig )
{
   loadParameters( iConfig );
}

