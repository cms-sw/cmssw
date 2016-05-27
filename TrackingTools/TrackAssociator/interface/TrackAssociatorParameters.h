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
//
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"

class TrackAssociatorParameters {
 public:
   TrackAssociatorParameters(){}
   TrackAssociatorParameters( const edm::ParameterSet&, edm::ConsumesCollector&& );
   void loadParameters( const edm::ParameterSet&, edm::ConsumesCollector& );
   
   double dREcal;
   double dRHcal;
   double dRMuon;
   
   double dREcalPreselection;
   double dRHcalPreselection;
   double dRMuonPreselection;
   double dRPreshowerPreselection;
   
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
   bool usePreshower;
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
   edm::InputTag theGEMSegmentCollectionLabel;
   
   // Specify if we want to widen the search pass of the crossed
   // calorimeter elements taking into account uncertainty
   // of the track trajectory. The parameter below
   // specifies how many standard deviations
   // to account for. Negative numbers are ignored
   // and trajectory is assumed to be known perfectly
   double trajectoryUncertaintyTolerance;

   edm::EDGetTokenT<EBRecHitCollection> EBRecHitsToken;
   edm::EDGetTokenT<EERecHitCollection> EERecHitsToken;
   edm::EDGetTokenT<CaloTowerCollection> caloTowersToken;
   edm::EDGetTokenT<HBHERecHitCollection> HBHEcollToken;
   edm::EDGetTokenT<HORecHitCollection> HOcollToken;
   edm::EDGetTokenT<DTRecSegment4DCollection> dtSegmentsToken;
   edm::EDGetTokenT<CSCSegmentCollection> cscSegmentsToken;
   edm::EDGetTokenT<GEMSegmentCollection> gemSegmentsToken;
   edm::EDGetTokenT<edm::SimTrackContainer> simTracksToken;
   edm::EDGetTokenT<edm::SimVertexContainer> simVerticesToken;
   edm::EDGetTokenT<edm::PCaloHitContainer> simEcalHitsEBToken;
   edm::EDGetTokenT<edm::PCaloHitContainer> simEcalHitsEEToken;
   edm::EDGetTokenT<edm::PCaloHitContainer> simHcalHitsToken;
};
#endif
