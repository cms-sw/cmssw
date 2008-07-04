#ifndef TrackAssociator_TrackAssociatorParameters_h
#define TrackAssociator_TrackAssociatorParameters_h 1

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
// $Id: TrackAssociatorParameters.h,v 1.4 2007/10/08 11:23:34 dmytro Exp $
//
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackAssociatorParameters {
 public:
   enum CrossedEnergyAlgorithmType { SinglePointAlongTrajectory, FivePointTwoSigmaElipseAlongTrajectory };
   TrackAssociatorParameters(){}
   TrackAssociatorParameters( const edm::ParameterSet& );
   void loadParameters( const edm::ParameterSet& );
   
   double dREcal;
   double dRHcal;
   double dRMuon;
   
   double dREcalPreselection;
   double dRHcalPreselection;
   double dRMuonPreselection;
   
   /// account for trajectory change for calorimeters.
   /// allows to compute energy around original track direction 
   /// (for example neutral particles in a jet) as well as energy
   /// around track projection on the inner surface of a 
   /// calorimeter. Affects performance, so use wisely.
   bool accountForTrajectoryChangeCalo;
   
   // account for trajectory change in the muon detector
   // helps to ensure that all chambers are found. 
   // Recomended to be used in default configuration
   // bool accountForTrajectoryChangeMuon;

   /// maximal distance from a muon chamber. Can be considered as a preselection
   /// cut and fancier cuts can be applied in a muon producer, since the
   /// distance from a chamber should be available as output of the TrackAssociation
   double muonMaxDistanceX;
   double muonMaxDistanceY;
   double muonMaxDistanceSigmaX;
   double muonMaxDistanceSigmaY;
   
   bool useEcal;
   bool useHcal;
   bool useHO;
   bool useCalo;
   bool useMuon;
   bool truthMatch;
   
   /// Labels of the detector EDProducts 
   edm::InputTag theEBRecHitCollectionLabel;
   edm::InputTag theEERecHitCollectionLabel;
   edm::InputTag theCaloTowerCollectionLabel;
   edm::InputTag theHBHERecHitCollectionLabel;
   edm::InputTag theHORecHitCollectionLabel;
   edm::InputTag theDTRecSegment4DCollectionLabel;
   edm::InputTag theCSCSegmentCollectionLabel;
   
   CrossedEnergyAlgorithmType crossedEnergyType;
   bool propagateAllDirections; // needs TrackExtra if set true
};
#endif
