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
// $Id: TrackDetectorAssociator.h,v 1.4 2007/02/19 12:02:41 dmytro Exp $
//
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackAssociatorParameters {
 public:
   TrackAssociatorParameters(){}
   TrackAssociatorParameters( const edm::ParameterSet& );
   void loadParameters( const edm::ParameterSet& );
   
   double dREcal;
   double dRHcal;
   double dRMuon;
   
   double dREcalPreselection;
   double dRHcalPreselection;
   double dRMuonPreselection;
   
   /// maximal distance from a muon chamber. Can be consider as a preselection
   /// cut and fancier cuts can be applied in a muon producer, since the
   /// distance from a chamber should be available as output of the TrackAssociation
   double muonMaxDistanceX;
   double muonMaxDistanceY;
   
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
   

};
#endif
