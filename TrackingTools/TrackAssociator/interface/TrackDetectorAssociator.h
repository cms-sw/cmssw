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
// $Id: TrackDetectorAssociator.h,v 1.3 2007/02/08 00:14:29 dmytro Exp $
//
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/OrphanHandle.h"

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/EcalDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/MuonDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/HcalDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/HODetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"
#include "TrackingTools/TrackAssociator/interface/CachedTrajectory.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/RecSegment.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

class TrackDetectorAssociator {
 public:
   TrackDetectorAssociator();
   ~TrackDetectorAssociator();
   
   class AssociatorParameters {
    public:
      ///track associator parameters
      AssociatorParameters() {
	 // default parameters
	 // define match cones, dR=sqrt(dEta^2+dPhi^2)
	 dREcal = 0.03;
	 dRHcal = 0.2;
	 dRMuon = 0.1;
	 dREcalPreselection = dREcal;
	 dRHcalPreselection = dRHcal;
	 dRMuonPreselection = dRMuon;
	 // match all sub-detectors by default
	 useEcal = true;
	 useHcal = true;
	 useCalo = true;
	 useHO = true;
	 useMuon = true;
	 useOldMuonMatching = false;
	 muonMaxDistanceX = 5;
	 muonMaxDistanceY = 5;
      }
      double dREcal;
      double dRHcal;
      double dRMuon;
      // should be used if it's expected to get a set of DetIds, when the
      // trajectory is not known yet and only a distant point with direction
      // is available. By default it is set to the final cuts
      double dREcalPreselection;
      double dRHcalPreselection;
      double dRMuonPreselection;
      // maximal distance from a muon chamber. Can be consider as a preselection
      // cut and fancier cuts can be applied in a muon producer, since the
      // distance from a chamber should be available as output of the TrackAssociation
      double muonMaxDistanceX;
      double muonMaxDistanceY;
      bool useEcal;
      bool useHcal;
      bool useHO;
      bool useCalo;
      bool useMuon;
      bool useOldMuonMatching;
   };
   
   
   /// propagate a track across the whole detector and
   /// find associated objects. Association is done in
   /// two modes 
   ///  1) an object is associated to a track only if 
   ///     crossed by track
   ///  2) an object is associated to a track if it is
   ///     withing an eta-phi cone of some radius with 
   ///     respect to a track.
   ///     (the cone origin is at (0,0,0))
   TrackDetMatchInfo            associate( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const AssociatorParameters& );

   /// associate ECAL only and return RecHits
   /// negative dR means only crossed elements
   std::vector<EcalRecHit>  associateEcal( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const double dR = -1 );
   
   /// associate ECAL only and return energy
   /// negative dR means only crossed elements
   double                   getEcalEnergy( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const double dR = -1 );
   
   /// associate ECAL only and return RecHits
   /// negative dR means only crossed elements
   std::vector<CaloTower>   associateHcal( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const double dR = -1 );

   /// associate ECAL only and return energy
   /// negative dR means only crossed elements
   double                   getHcalEnergy( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const double dR = -1 );
   /// use a user configured propagator
   void setPropagator( Propagator* );
   
   /// use the default propagator
   void useDefaultPropagator();
   
   /// get FreeTrajectoryState from different track representations
   FreeTrajectoryState getFreeTrajectoryState( const edm::EventSetup&, 
					       const reco::Track& );
   FreeTrajectoryState getFreeTrajectoryState( const edm::EventSetup&, 
					       const SimTrack&, 
					       const SimVertex& );
   /// Labels of the detector EDProducts 
   edm::InputTag theEBRecHitCollectionLabel;
   edm::InputTag theEERecHitCollectionLabel;
   edm::InputTag theCaloTowerCollectionLabel;
   edm::InputTag theHBHERecHitCollectionLabel;
   edm::InputTag theHORecHitCollectionLabel;
   edm::InputTag theDTRecSegment4DCollectionLabel;
   edm::InputTag theCSCSegmentCollectionLabel;
   
 private:
   void       fillEcal( const edm::Event&,
			TrackDetMatchInfo&, 
			const AssociatorParameters&);
   
   void fillCaloTowers( const edm::Event&,
			TrackDetMatchInfo&,
			const AssociatorParameters&);
   
   void       fillHcal( const edm::Event&,
			TrackDetMatchInfo&,
			const AssociatorParameters&);
   
   void         fillHO( const edm::Event&,
			TrackDetMatchInfo&,
			const AssociatorParameters&);
  
   void fillMuon(       const edm::Event&,
			TrackDetMatchInfo&,
			const AssociatorParameters&);
   
   void addMuonSegmentMatch(MuonChamberMatch&,
			    const RecSegment*,
			    const AssociatorParameters&);
   
   void getMuonChamberMatches(std::vector<MuonChamberMatch>& matches,
			      const float dRMuonPreselection,
			      const float maxDistanceX,
			      const float maxDistanceY);
  
   void           init( const edm::EventSetup&);
   
   math::XYZPoint getPoint( const GlobalPoint& point)
     {
	return math::XYZPoint(point.x(),point.y(),point.z());
     }
   
   math::XYZPoint getPoint( const LocalPoint& point)
     {
	return math::XYZPoint(point.x(),point.y(),point.z());
     }
   
   math::XYZVector getVector( const GlobalVector& vec)
     {
	return math::XYZVector(vec.x(),vec.y(),vec.z());
     }
   
   math::XYZVector getVector( const LocalVector& vec)
     {
	return math::XYZVector(vec.x(),vec.y(),vec.z());
     }
   
   Propagator* ivProp_;
   Propagator* defProp_;
   CachedTrajectory cachedTrajectory_;
   bool useDefaultPropagator_;
   
   EcalDetIdAssociator ecalDetIdAssociator_;
   HcalDetIdAssociator hcalDetIdAssociator_;
   HODetIdAssociator   hoDetIdAssociator_;
   CaloDetIdAssociator caloDetIdAssociator_;
   MuonDetIdAssociator muonDetIdAssociator_;
   
   edm::ESHandle<CaloGeometry> theCaloGeometry_;
   edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry_;
};
#endif
