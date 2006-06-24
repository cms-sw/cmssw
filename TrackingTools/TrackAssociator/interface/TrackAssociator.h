#ifndef TrackAssociator_TrackAssociator_h
#define TrackAssociator_TrackAssociator_h 1

// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      TrackAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: TrackAssociator.h,v 1.1 2006/06/09 17:30:20 dmytro Exp $
//
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/OrphanHandle.h"

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/EcalDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "SimDataFormats/Track/interface/EmbdSimTrack.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertex.h"

class TrackAssociator {
 public:
   TrackAssociator();
   ~TrackAssociator();
   
   class AssociatorParameters {
    public:
      AssociatorParameters() {
	 // default parameters
	 // define match cones, dR=sqrt(dEta^2+dPhi^2)
	 dREcal = 0.03;
	 dRHcal = 0.07;
	 dRMuon = 0.1;
	 // match all sub-detectors by default
	 useEcal = true;
	 useHcal = true;
	 useMuon = true;
      }
      double dREcal;
      double dRHcal;
      double dRMuon;
      bool useEcal;
      bool useHcal;
      bool useMuon;
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
   
   /// specify names of EDProducts to use for different input data types
   void addDataLabels( const std::string className,
		       const std::string moduleLabel,
		       const std::string productInstanceLabel = "");
   
   /// get FreeTrajectoryState from different track representations
   FreeTrajectoryState getFreeTrajectoryState( const edm::EventSetup&, 
					       const reco::Track& );
   FreeTrajectoryState getFreeTrajectoryState( const edm::EventSetup&, 
					       const EmbdSimTrack&, 
					       const EmbdSimVertex& );
   
 private:
   void       fillEcal( const edm::Event&,
			const edm::EventSetup&,
			TrackDetMatchInfo&, 
			const FreeTrajectoryState&,
			const double);
   
   void fillCaloTowers( const edm::Event&,
			const edm::EventSetup&,
			TrackDetMatchInfo&, 
			const FreeTrajectoryState&,
			const double);
   
   void fillDTSegments( const edm::Event&,
			const edm::EventSetup&,
			TrackDetMatchInfo&,
			const FreeTrajectoryState&,
			const double);
  
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
   bool useDefaultPropagator_;
   int debug_;
   std::vector<std::vector<std::set<uint32_t> > >* caloTowerMap_;
   
   EcalDetIdAssociator ecalDetIdAssociator_;
   CaloDetIdAssociator caloDetIdAssociator_;
   
   edm::ESHandle<CaloGeometry> theCaloGeometry_;
   
   /// Labels of the detector EDProducts (empty by default)
   /// ECAL
   std::vector<std::string> EBRecHitCollectionLabels;
   /// CaloTowers
   std::vector<std::string> CaloTowerCollectionLabels;
   /// Muons
   std::vector<std::string> DTRecSegment4DCollectionLabels;
};
#endif
