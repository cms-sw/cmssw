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
// $Id: TrackAssociator.cc,v 1.3 2006/08/09 14:44:48 dmytro Exp $
//
//

#include "TrackingTools/TrackAssociator/interface/TrackAssociator.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/OrphanHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

// calorimeter info
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "Geometry/Surface/interface/Cylinder.h"
#include "Geometry/Surface/interface/Plane.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "Utilities/Timing/interface/TimingReport.h"
#include <stack>
#include <set>


#include "TrackingTools/TrackAssociator/interface/CaloDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/EcalDetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"

//
// class declaration
//

using namespace reco;

TrackAssociator::TrackAssociator() 
{
   ivProp_ = 0;
   defProp_ = 0;
   debug_ = 0;
   caloTowerMap_ = 0;
   useDefaultPropagator_ = false;
}

TrackAssociator::~TrackAssociator()
{
   if (defProp_) delete defProp_;
}

void TrackAssociator::addDataLabels( const std::string className,
				     const std::string moduleLabel,
				     const std::string productInstanceLabel)
{
   if (className == "EBRecHitCollection")
     {
	EBRecHitCollectionLabels.clear();
	EBRecHitCollectionLabels.push_back(moduleLabel);
	EBRecHitCollectionLabels.push_back(productInstanceLabel);
     }
   if (className == "CaloTowerCollection")
     {
	CaloTowerCollectionLabels.clear();
	CaloTowerCollectionLabels.push_back(moduleLabel);
	CaloTowerCollectionLabels.push_back(productInstanceLabel);
     }
   if (className == "DTRecSegment4DCollection")
     {
	DTRecSegment4DCollectionLabels.clear();
	DTRecSegment4DCollectionLabels.push_back(moduleLabel);
	DTRecSegment4DCollectionLabels.push_back(productInstanceLabel);
     }
}


void TrackAssociator::setPropagator( Propagator* ptr)
{
   ivProp_ = ptr; 
   caloDetIdAssociator_.setPropagator(ivProp_);
   ecalDetIdAssociator_.setPropagator(ivProp_);
}

void TrackAssociator::useDefaultPropagator()
{
   useDefaultPropagator_ = true;
}


void TrackAssociator::init( const edm::EventSetup& iSetup )
{
   // access the calorimeter geometry
   iSetup.get<IdealGeometryRecord>().get(theCaloGeometry_);
   if (!theCaloGeometry_.isValid()) 
     throw cms::Exception("FatalError") << "Unable to find IdealGeometryRecord in event!\n";
   
   if (useDefaultPropagator_ && ! defProp_ ) {
      // setup propagator
      edm::ESHandle<MagneticField> bField;
      iSetup.get<IdealMagneticFieldRecord>().get(bField);
      
      SteppingHelixPropagator* prop  = new SteppingHelixPropagator(&*bField,anyDirection);
      prop->setMaterialMode(false);
      prop->applyRadX0Correction(true);
      // prop->setDebug(true); // tmp
      defProp_ = prop;
      setPropagator(defProp_);
   }
   
	
}

TrackDetMatchInfo TrackAssociator::associate( const edm::Event& iEvent,
					      const edm::EventSetup& iSetup,
					      const FreeTrajectoryState& trackOrigin,
					      const AssociatorParameters& parameters )
{
   TrackDetMatchInfo info;
   using namespace edm;
   TimerStack timers;
   
   init( iSetup );
   
   FreeTrajectoryState currentPosition(trackOrigin);

   if (parameters.useEcal) fillEcal( iEvent, iSetup, info, currentPosition, parameters.dREcal);
   if (parameters.useHcal) fillCaloTowers( iEvent, iSetup, info, currentPosition, parameters.dRHcal);
   if (parameters.useMuon) fillDTSegments( iEvent, iSetup, info, currentPosition, parameters.dRMuon);
   
   return info;
}



std::vector<EcalRecHit> TrackAssociator::associateEcal( const edm::Event& iEvent,
							const edm::EventSetup& iSetup,
							const FreeTrajectoryState& trackOrigin,
							const double dR )
{
   AssociatorParameters parameters;
   parameters.useHcal = false;
   parameters.useMuon = false;
   parameters.dREcal = dR;
   TrackDetMatchInfo info( associate(iEvent, iSetup, trackOrigin, parameters ));
   if (dR>0) 
     return info.ecalRecHits;
   else
     return info.crossedEcalRecHits;
}

double TrackAssociator::getEcalEnergy( const edm::Event& iEvent,
				       const edm::EventSetup& iSetup,
				       const FreeTrajectoryState& trackOrigin,
				       const double dR )
{
   AssociatorParameters parameters;
   parameters.useHcal = false;
   parameters.useMuon = false;
   parameters.dREcal = dR;
   TrackDetMatchInfo info = associate(iEvent, iSetup, trackOrigin, parameters );
   if(dR>0) 
     return info.ecalConeEnergy();
   else
     return info.ecalEnergy();
}

std::vector<CaloTower> TrackAssociator::associateHcal( const edm::Event& iEvent,
						       const edm::EventSetup& iSetup,
						       const FreeTrajectoryState& trackOrigin,
						       const double dR )
{
   AssociatorParameters parameters;
   parameters.useEcal = false;
   parameters.useMuon = false;
   parameters.dRHcal = dR;
   TrackDetMatchInfo info( associate(iEvent, iSetup, trackOrigin, parameters ));
   if (dR>0) 
     return info.towers;
   else
     return info.crossedTowers;
   
}

double TrackAssociator::getHcalEnergy( const edm::Event& iEvent,
				       const edm::EventSetup& iSetup,
				       const FreeTrajectoryState& trackOrigin,
				       const double dR )
{
   AssociatorParameters parameters;
   parameters.useEcal = false;
   parameters.useMuon = false;
   parameters.dRHcal = dR;
   TrackDetMatchInfo info( associate(iEvent, iSetup, trackOrigin, parameters ));
   if (dR>0) 
     return info.hcalConeEnergy();
   else
     return info.hcalEnergy();
}


void TrackAssociator::fillEcal( const edm::Event& iEvent,
				const edm::EventSetup& iSetup,
				TrackDetMatchInfo& info,
				const FreeTrajectoryState& trajectoryPoint,
				const double dR)
{
   TimerStack timers;
   timers.push("TrackAssociator::fillEcal");
   
   ecalDetIdAssociator_.setGeometry(&*theCaloGeometry_);
   
   timers.push("TrackAssociator::fillEcal::propagation");
   // ECAL points (EB+EE)
   std::vector<GlobalPoint> ecalPoints;
   ecalPoints.push_back(GlobalPoint(135.,0,310.));
   ecalPoints.push_back(GlobalPoint(150.,0,340.));
   ecalPoints.push_back(GlobalPoint(170.,0,370.));
   
   std::vector<GlobalPoint> ecalTrajectory = ecalDetIdAssociator_.getTrajectory(trajectoryPoint, ecalPoints);
   if(ecalTrajectory.empty()) throw cms::Exception("FatalError") << "Failed to propagate a track to ECAL\n";
   info.trkGlobPosAtEcal = getPoint(ecalTrajectory[0]);

   // Find ECAL crystals
   timers.pop_and_push("TrackAssociator::fillEcal::access::EcalBarrel");
   edm::Handle<EBRecHitCollection> EBRecHits;
   if (EBRecHitCollectionLabels.empty())
     // iEvent_->getByType (EBRecHits);
     throw cms::Exception("FatalError") << "Module lable is not set for EBRecHitCollection.\n";
   else
     iEvent.getByLabel (EBRecHitCollectionLabels[0], EBRecHitCollectionLabels[1], EBRecHits);
   if (!EBRecHits.isValid()) throw cms::Exception("FatalError") << "Unable to find EBRecHitCollection in event!\n";

   timers.pop_and_push("TrackAssociator::fillEcal::matching");
   std::set<DetId> ecalIdsInRegion = ecalDetIdAssociator_.getDetIdsCloseToAPoint(ecalTrajectory[0],dR);
   // std::cout << "ecalIdsInRegion.size(): " << ecalIdsInRegion.size() << std::endl;
   std::set<DetId> ecalIdsInACone =  ecalDetIdAssociator_.getDetIdsInACone(ecalIdsInRegion, ecalTrajectory, dR);
   // std::cout << "ecalIdsInACone.size(): " << ecalIdsInACone.size() << std::endl;
   std::set<DetId> crossedEcalIds =  ecalDetIdAssociator_.getCrossedDetIds(ecalIdsInRegion, ecalTrajectory);
   // std::cout << "crossedEcalIds.size(): " << crossedEcalIds.size() << std::endl;
   
   // add EcalRecHits
   timers.pop_and_push("TrackAssociator::fillEcal::addEcalRecHits");
   for(std::set<DetId>::const_iterator itr=crossedEcalIds.begin(); itr!=crossedEcalIds.end();itr++)
     {
	std::vector<EcalRecHit>::const_iterator hit = (*EBRecHits).find(*itr);
	if(hit != (*EBRecHits).end()) 
	  info.crossedEcalRecHits.push_back(*hit);
	else  
	   LogDebug("TrackAssociator::fillEcal") << "EcalRecHit is not found for DetId: " << itr->rawId() <<"\n";
     }
   for(std::set<DetId>::const_iterator itr=ecalIdsInACone.begin(); itr!=ecalIdsInACone.end();itr++)
     {
	std::vector<EcalRecHit>::const_iterator hit = (*EBRecHits).find(*itr);
	if(hit != (*EBRecHits).end()) 
	  info.ecalRecHits.push_back(*hit);
	else 
	  LogDebug("TrackAssociator::fillEcal") << "EcalRecHit is not found for DetId: " << itr->rawId() <<"\n";
     }
}

void TrackAssociator::fillCaloTowers( const edm::Event& iEvent,
				      const edm::EventSetup& iSetup,
				      TrackDetMatchInfo& info,
				      const FreeTrajectoryState& trajectoryPoint,
				      const double dR)
{
   // ECAL hits are not used for the CaloTower identification
   TimerStack timers;
   timers.push("TrackAssociator::fillCaloTowers");

   caloDetIdAssociator_.setGeometry(&*theCaloGeometry_);
   
   // HCAL points (HB+HE)
   timers.push("TrackAssociator::fillCaloTowers::propagation");
   std::vector<GlobalPoint> hcalPoints;
   hcalPoints.push_back(GlobalPoint(190.,0,400.));
   hcalPoints.push_back(GlobalPoint(240.,0,500.));
   hcalPoints.push_back(GlobalPoint(280.,0,550.));
   
   std::vector<GlobalPoint> hcalTrajectory = caloDetIdAssociator_.getTrajectory(trajectoryPoint, hcalPoints);
   if(hcalTrajectory.empty()) throw cms::Exception("FatalError") << "Failed to propagate the track to HCAL\n";
   info.trkGlobPosAtHcal = getPoint(hcalTrajectory[0]);
   
   // find crossed CaloTowers
   timers.pop_and_push("TrackAssociator::fillCaloTowers::access::CaloTowers");
   edm::Handle<CaloTowerCollection> caloTowers;

   if (CaloTowerCollectionLabels.empty())
     // iEvent_->getByType (caloTowers);
     throw cms::Exception("FatalError") << "Module lable is not set for CaloTowers.\n";
   else
     iEvent.getByLabel (CaloTowerCollectionLabels[0], CaloTowerCollectionLabels[1], caloTowers);
   if (!caloTowers.isValid())  throw cms::Exception("FatalError") << "Unable to find CaloTowers in event!\n";
   
   timers.push("TrackAssociator::fillCaloTowers::matching");
   std::set<DetId> caloTowerIdsInRegion = caloDetIdAssociator_.getDetIdsCloseToAPoint(hcalTrajectory[0],dR);
   std::set<DetId> caloTowerIdsInACone = caloDetIdAssociator_.getDetIdsInACone(caloTowerIdsInRegion, hcalTrajectory, dR);
   std::set<DetId> crossedCaloTowerIds = caloDetIdAssociator_.getCrossedDetIds(caloTowerIdsInRegion, hcalTrajectory);
   
   // add CaloTowers
   timers.push("TrackAssociator::fillCaloTowers::addCaloTowers");
   for(std::set<DetId>::const_iterator itr=crossedCaloTowerIds.begin(); itr!=crossedCaloTowerIds.end();itr++)
     {
	DetId id(*itr);
	CaloTowerCollection::const_iterator tower = (*caloTowers).find(id);
	if(tower != (*caloTowers).end()) 
	  info.crossedTowers.push_back(*tower);
	else
	  LogDebug("TrackAssociator::fillEcal") << "CaloTower is not found for DetId: " << id.rawId() << "\n";
     }

   for(std::set<DetId>::const_iterator itr=caloTowerIdsInACone.begin(); itr!=caloTowerIdsInACone.end();itr++)
     {
	DetId id(*itr);
	CaloTowerCollection::const_iterator tower = (*caloTowers).find(id);
	if(tower != (*caloTowers).end()) 
	  info.towers.push_back(*tower);
	else 
	  LogDebug("TrackAssociator::fillEcal") << "CaloTower is not found for DetId: " << id.rawId() << "\n";
     }
   
}

void TrackAssociator::fillDTSegments( const edm::Event& iEvent,
				      const edm::EventSetup& iSetup,
				      TrackDetMatchInfo& info,
				      const FreeTrajectoryState& trajectoryPoint,
				      const double dR)
{
   TimerStack timers;
   timers.push("TrackAssociator::fillDTSegments");
   using namespace edm;
   TrajectoryStateOnSurface tSOSDest;
   
   // Get the DT Geometry
   ESHandle<DTGeometry> dtGeom;
   iSetup.get<MuonGeometryRecord>().get(dtGeom);

   // Get the rechit collection from the event
   timers.push("TrackAssociator::fillDTSegments::access");
   Handle<DTRecSegment4DCollection> dtSegments;
   if (DTRecSegment4DCollectionLabels.empty())
     // iEvent_->getByType (dtSegments);
     throw cms::Exception("FatalError") << "Module lable is not set for DTRecSegment4DCollection.\n";
   else
     iEvent.getByLabel (DTRecSegment4DCollectionLabels[0], DTRecSegment4DCollectionLabels[1], dtSegments);
   if (! dtSegments.isValid()) 
     throw cms::Exception("FatalError") << "Unable to find DTRecSegment4DCollection in event!\n";

   // Iterate over all detunits
   DTRecSegment4DCollection::id_iterator detUnitIt;
   for (detUnitIt = dtSegments->id_begin();
	detUnitIt != dtSegments->id_end();
	++detUnitIt){
      // Get the GeomDet from the setup
      const DTChamber* chamber = dtGeom->chamber(*detUnitIt);
      if (chamber == 0){
	std::cout<<"Failed to get detector unit"<<std::endl;
	continue;
      }
      const Surface& surf = chamber->surface();
      
      if (debug_){
	 std::cout<<"Will propagate to surface: "<<surf.position()<<" "<<surf.rotation()<<std::endl;
      }
      tSOSDest = ivProp_->Propagator::propagate(trajectoryPoint, surf);
         
      // Get the range for the corresponding LayerId
      DTRecSegment4DCollection::range  range = dtSegments->get((*detUnitIt));
      // Loop over the rechits of this DetUnit
      for (DTRecSegment4DCollection::const_iterator recseg = range.first;
	   recseg!=range.second;
	   recseg++){
	 
	 LogDebug("TrackAssociator::fillDTSegments")
	   << "Segment local position: " << recseg->localPosition() << "\n"
	   << std::hex << recseg->geographicalId().rawId() << "\n";
	 
	 GlobalPoint dtSeg = surf.toGlobal(recseg->localPosition());
	 
	 LogDebug("TrackAssociator::fillDTSegments")
	   << "Segment global position: " << dtSeg << " \t (R_xy,eta,phi): "
	   << dtSeg.perp() << "," << dtSeg.eta() << "," << dtSeg.phi() << "\n";
	 
	 LogDebug("TrackAssociator::fillDTSegments")
	   << "\teta hit: " << dtSeg.eta() << " \tpropagator: " << tSOSDest.freeState()->position().eta() << "\n"
	   << "\tphi hit: " << dtSeg.phi() << " \tpropagator: " << tSOSDest.freeState()->position().phi() << std::endl;
	 
	 if (sqrt( pow(dtSeg.eta()-tSOSDest.freeState()->position().eta(),2) + 
		   pow(dtSeg.phi()-tSOSDest.freeState()->position().phi(),2) ) < dR)
	   {
	      MuonSegmentMatch muonSegment;
	      muonSegment.segmentGlobalPosition = getPoint(dtSeg);
	      muonSegment.segmentLocalPosition = getPoint( recseg->localPosition() );
	      muonSegment.segmentLocalDirection = getVector( recseg->localDirection() );
	      muonSegment.segmentLocalErrorXX = recseg->localPositionError().xx();
	      muonSegment.segmentLocalErrorYY = recseg->localPositionError().yy();
	      muonSegment.segmentLocalErrorXY = recseg->localPositionError().xy();
	      muonSegment.segmentLocalErrorDxDz = recseg->localDirectionError().xx();
	      muonSegment.segmentLocalErrorDyDz = recseg->localDirectionError().yy();
	      
	      // muon.segmentPosition_.push_back(getPoint(dtSeg));
	      muonSegment.trajectoryGlobalPosition = getPoint(tSOSDest.freeState()->position()) ;
	      muonSegment.trajectoryLocalPosition = getPoint(surf.toLocal(tSOSDest.freeState()->position()));
	      muonSegment.trajectoryLocalDirection = getVector(surf.toLocal(tSOSDest.freeState()->momentum()));
	      // muon.trajectoryDirection_.push_back(getVector(tSOSDest.freeState()->momentum()));
	      float errXX(-1.), errYY(-1.), errXY(-1.);
	      float err_dXdZ(-1.), err_dYdZ(-1.);
	      if (tSOSDest.freeState()->hasError()){
		 LocalError err = ErrorFrameTransformer().transform( tSOSDest.freeState()->cartesianError().position(), surf );
		 errXX = err.xx();
		 errXY = err.xy();
		 errYY = err.yy();
	      }
	      muonSegment.trajectoryLocalErrorXX = errXX;
	      muonSegment.trajectoryLocalErrorXY = errXY;
	      muonSegment.trajectoryLocalErrorYY = errYY;
	      muonSegment.trajectoryLocalErrorDxDz = err_dXdZ;
	      muonSegment.trajectoryLocalErrorDyDz = err_dYdZ;
	      muonSegment.id = (*detUnitIt).rawId();
	      info.segments.push_back(muonSegment);
	      
		  // Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h
		  // LocalError transform(const GlobalError& ge, const Surface& surf)
	   }
      }
   }
}


FreeTrajectoryState TrackAssociator::getFreeTrajectoryState( const edm::EventSetup& iSetup, 
							     const SimTrack& track, 
							     const SimVertex& vertex )
{
   edm::ESHandle<MagneticField> bField;
   iSetup.get<IdealMagneticFieldRecord>().get(bField);
   
   GlobalVector vector( track.momentum().x(), track.momentum().y(), track.momentum().z() );
   // convert mm to cm
   GlobalPoint point( vertex.position().x()*.1, vertex.position().y()*.1, vertex.position().z()*.1 );
   int charge = track.type( )> 0 ? -1 : 1;
   GlobalTrajectoryParameters tPars(point, vector, charge, &*bField);
   
   HepSymMatrix covT(6,1); covT *= 1e-6; // initialize to sigma=1e-3
   CartesianTrajectoryError tCov(covT);
   
   return FreeTrajectoryState(tPars, tCov);
}


FreeTrajectoryState TrackAssociator::getFreeTrajectoryState( const edm::EventSetup& iSetup,
							     const reco::Track& track )
{
   edm::ESHandle<MagneticField> bField;
   iSetup.get<IdealMagneticFieldRecord>().get(bField);
   
   GlobalVector vector( track.momentum().x(), track.momentum().y(), track.momentum().z() );

   GlobalPoint point( track.vertex().x(), track.vertex().y(),  track.vertex().z() );

   GlobalTrajectoryParameters tPars(point, vector, track.charge(), &*bField);
   
   // FIX THIS !!!
   // need to convert from perigee to global or helix (curvilinear) frame
   // for now just an arbitrary matrix.
   HepSymMatrix covT(6,1); covT *= 1e-6; // initialize to sigma=1e-3
   CartesianTrajectoryError tCov(covT);
   
   return FreeTrajectoryState(tPars, tCov);
}

