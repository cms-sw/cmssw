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
// $Id: TrackDetectorAssociator.h,v 1.9 2007/05/16 09:26:23 dmytro Exp $
//
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

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

#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

class TrackDetectorAssociator {
 public:
   TrackDetectorAssociator();
   ~TrackDetectorAssociator();
   
   typedef TrackAssociatorParameters AssociatorParameters;
   
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
   TrackDetMatchInfo            associate( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const AssociatorParameters& );
   /// associate using inner and outer most states of a track
   /// in the silicon tracker. 
   TrackDetMatchInfo            associate( const edm::Event& iEvent,
					   const edm::EventSetup& iSetup,
					   const AssociatorParameters& parameters,
					   const FreeTrajectoryState* innerState,
					   const FreeTrajectoryState* outerState=0);
   /// associate using reco::Track
   TrackDetMatchInfo            associate( const edm::Event&,
					   const edm::EventSetup&,
					   const reco::Track&,
					   const AssociatorParameters& );
   /// associate using a simulated track
   TrackDetMatchInfo            associate( const edm::Event&,
					   const edm::EventSetup&,
					   const SimTrack&,
					   const SimVertex&,
					   const AssociatorParameters& );
   /// associate using 3-momentum, vertex and charge
   TrackDetMatchInfo            associate( const edm::Event&,
					   const edm::EventSetup&,
					   const GlobalVector&,
					   const GlobalPoint&,
					   const int,
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
   void setPropagator( const Propagator* );
   
   /// use the default propagator
   void useDefaultPropagator();
   
   /// get FreeTrajectoryState from different track representations
   static FreeTrajectoryState getFreeTrajectoryState( const edm::EventSetup&, 
						      const reco::Track& );
   static FreeTrajectoryState getFreeTrajectoryState( const edm::EventSetup&, 
						      const SimTrack&, 
						      const SimVertex& );
   static FreeTrajectoryState getFreeTrajectoryState( const edm::EventSetup&,
						      const GlobalVector&,
						      const GlobalPoint&,
						      const int);
 private:
   DetIdAssociator::MapRange getMapRange( const std::pair<float,float>& delta,
					  const float dR );

   void fillEcal(       const edm::Event&,
			TrackDetMatchInfo&, 
			const AssociatorParameters&);
   
   void fillCaloTowers( const edm::Event&,
			TrackDetMatchInfo&,
			const AssociatorParameters&);
   
   void fillHcal(       const edm::Event&,
			TrackDetMatchInfo&,
			const AssociatorParameters&);
   
   void fillHO(         const edm::Event&,
			TrackDetMatchInfo&,
			const AssociatorParameters&);
  
   void fillMuon(       const edm::Event&,
			TrackDetMatchInfo&,
			const AssociatorParameters&);
   
   void fillCaloTruth(  const edm::Event&,
			TrackDetMatchInfo&,
			const AssociatorParameters&);
   
   void addMuonSegmentMatch( ::MuonChamberMatch&,
			    const RecSegment*,
			    const AssociatorParameters&);
   
   void getMuonChamberMatches(std::vector< ::MuonChamberMatch>& matches,
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
   
   const Propagator* ivProp_;
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
