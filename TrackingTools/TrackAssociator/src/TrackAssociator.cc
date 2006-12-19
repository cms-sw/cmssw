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
// $Id: TrackAssociator.cc,v 1.14 2006/10/20 16:37:44 dmytro Exp $
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
#include "DataFormats/DetId/interface/DetId.h"

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

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

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
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "Geometry/CommonDetAlgo/interface/ErrorFrameTransformer.h"

#include "CLHEP/HepPDT/ParticleID.hh"
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
   if (className == "EERecHitCollection")
     {
	EERecHitCollectionLabels.clear();
	EERecHitCollectionLabels.push_back(moduleLabel);
	EERecHitCollectionLabels.push_back(productInstanceLabel);
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
   if (className == "CSCSegmentCollection")
     {
	CSCSegmentCollectionLabels.clear();
	CSCSegmentCollectionLabels.push_back(moduleLabel);
	CSCSegmentCollectionLabels.push_back(productInstanceLabel);
     }
}


void TrackAssociator::setPropagator( Propagator* ptr)
{
   ivProp_ = ptr; 
   caloDetIdAssociator_.setPropagator(ivProp_);
   ecalDetIdAssociator_.setPropagator(ivProp_);
   muonDetIdAssociator_.setPropagator(ivProp_);
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
   
   // get the tracking Geometry
   iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry_);
   if (!theTrackingGeometry_.isValid()) 
     throw cms::Exception("FatalError") << "Unable to find GlobalTrackingGeometryRecord in event!\n";
   
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
   if (parameters.useMuon) {
      if (parameters.useOldMuonMatching){
	 fillDTSegments( iEvent, iSetup, info, currentPosition, parameters);
	 fillCSCSegments( iEvent, iSetup, info, currentPosition, parameters);
      }else{
	 fillMuonSegments( iEvent, iSetup, info, currentPosition, parameters);
      }
   }

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
				FreeTrajectoryState& trajectoryPoint,
				const double dR)
{
   TimerStack timers;
   timers.push("TrackAssociator::fillEcal");
   
   ecalDetIdAssociator_.setGeometry(&*theCaloGeometry_);
   
   timers.push("TrackAssociator::fillEcal::propagation");
   // ECAL points (EB+EE)
   // If the phi angle between a track entrance and exit points is more
   // than 2 crystals, it is possible that the track will cross 3 crystals
   // and therefore one has to check at least 3 points along the track
   // trajectory inside ECAL. In order to have a chance to cross 4 crystalls
   // in the barrel, a track should have P_t as low as 3 GeV or smaller
   // If it's necessary, number of points along trajectory can be increased
   std::vector<GlobalPoint> ecalPoints;
   ecalPoints.push_back(GlobalPoint(135.,0,315.));
   ecalPoints.push_back(GlobalPoint(140.,0,325.));
   ecalPoints.push_back(GlobalPoint(150.,0,335.));
   
   std::vector<GlobalPoint> ecalTrajectory = ecalDetIdAssociator_.getTrajectory(trajectoryPoint, ecalPoints);
   if(ecalTrajectory.empty()) {
      LogTrace("TrackAssociator::fillEcal") << "Failed to propagate a track to ECAL; moving on\n";
      info.isGoodEcal = 0;
      return;
   }
   info.isGoodEcal = 1;
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

   timers.pop_and_push("TrackAssociator::fillEcal::access::EcalEndcaps");
   edm::Handle<EERecHitCollection> EERecHits;
   if (EERecHitCollectionLabels.empty())
     // iEvent_->getByType (EERecHits);
     throw cms::Exception("FatalError") << "Module lable is not set for EERecHitCollection.\n";
   else
     iEvent.getByLabel (EERecHitCollectionLabels[0], EERecHitCollectionLabels[1], EERecHits);
   if (!EERecHits.isValid()) throw cms::Exception("FatalError") << "Unable to find EERecHitCollection in event!\n";

   timers.pop_and_push("TrackAssociator::fillEcal::matching");
   std::set<DetId> ecalIdsInRegion = ecalDetIdAssociator_.getDetIdsCloseToAPoint(ecalTrajectory[0],dR);
   LogTrace("TrackAssociator::fillEcal::matching") << "ecalIdsInRegion.size(): " << ecalIdsInRegion.size();
   std::set<DetId> ecalIdsInACone =  ecalDetIdAssociator_.getDetIdsInACone(ecalIdsInRegion, ecalTrajectory, dR);
   LogTrace("TrackAssociator::fillEcal::matching") << "ecalIdsInACone.size(): " << ecalIdsInACone.size();
   std::set<DetId> crossedEcalIds =  ecalDetIdAssociator_.getCrossedDetIds(ecalIdsInRegion, ecalTrajectory);
   LogTrace("TrackAssociator::fillEcal::matching") << "crossedEcalIds.size(): " << crossedEcalIds.size();
   
   // add EcalRecHits
   timers.pop_and_push("TrackAssociator::fillEcal::addEcalRecHits");
   for(std::set<DetId>::const_iterator itr=crossedEcalIds.begin(); itr!=crossedEcalIds.end();itr++)
   {
      std::vector<EcalRecHit>::const_iterator ebHit = (*EBRecHits).find(*itr);
      std::vector<EcalRecHit>::const_iterator eeHit = (*EERecHits).find(*itr);
      if(ebHit != (*EBRecHits).end()) 
         info.crossedEcalRecHits.push_back(*ebHit);
      else if(eeHit != (*EERecHits).end()) 
         info.crossedEcalRecHits.push_back(*eeHit);
      else  
         LogTrace("TrackAssociator::fillEcal") << "Crossed EcalRecHit is not found for DetId: " << itr->rawId();
   }
   for(std::set<DetId>::const_iterator itr=ecalIdsInACone.begin(); itr!=ecalIdsInACone.end();itr++)
   {
      std::vector<EcalRecHit>::const_iterator ebHit = (*EBRecHits).find(*itr);
      std::vector<EcalRecHit>::const_iterator eeHit = (*EERecHits).find(*itr);
      if(ebHit != (*EBRecHits).end()) 
         info.ecalRecHits.push_back(*ebHit);
      else if(eeHit != (*EERecHits).end()) 
         info.ecalRecHits.push_back(*eeHit);
      else 
         LogTrace("TrackAssociator::fillEcal") << "EcalRecHit from the cone is not found for DetId: " << itr->rawId();
   }
}

void TrackAssociator::fillCaloTowers( const edm::Event& iEvent,
				      const edm::EventSetup& iSetup,
				      TrackDetMatchInfo& info,
				      FreeTrajectoryState& trajectoryPoint,
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
   
   // get trajectory assuming simple geometry between barrel and endcap regions.
   std::vector<GlobalPoint> hcalTrajectory = caloDetIdAssociator_.getTrajectory(trajectoryPoint, hcalPoints, -1.);
   if(hcalTrajectory.empty()) {
      LogTrace("TrackAssociator::fillCaloTowers") << "Failed to propagate the track to HCAL; moving on\n";
      info.isGoodHcal = 0;
      return;
   }
   info.isGoodHcal = 1;
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
   LogTrace("TrackAssociator::fillHcal::matching") << "caloTowerIdsInRegion.size(): " << caloTowerIdsInRegion.size();
   std::set<DetId> caloTowerIdsInACone = caloDetIdAssociator_.getDetIdsInACone(caloTowerIdsInRegion, hcalTrajectory, dR);
   LogTrace("TrackAssociator::fillHcal::matching") << "caloTowerIdsInACone.size(): " << caloTowerIdsInACone.size();
   std::set<DetId> crossedCaloTowerIds = caloDetIdAssociator_.getCrossedDetIds(caloTowerIdsInRegion, hcalTrajectory);
   LogTrace("TrackAssociator::fillHcal::matching") << "crossedCaloTowerIds.size(): " << crossedCaloTowerIds.size();
   
   // add CaloTowers
   timers.push("TrackAssociator::fillCaloTowers::addCaloTowers");
   for(std::set<DetId>::const_iterator itr=crossedCaloTowerIds.begin(); itr!=crossedCaloTowerIds.end();itr++)
     {
	DetId id(*itr);
	CaloTowerCollection::const_iterator tower = (*caloTowers).find(id);
	if(tower != (*caloTowers).end()) 
	  info.crossedTowers.push_back(*tower);
	else
	  LogTrace("TrackAssociator::fillEcal") << "Crossed CaloTower is not found for DetId: " << id.rawId();
     }

   for(std::set<DetId>::const_iterator itr=caloTowerIdsInACone.begin(); itr!=caloTowerIdsInACone.end();itr++)
     {
	DetId id(*itr);
	CaloTowerCollection::const_iterator tower = (*caloTowers).find(id);
	if(tower != (*caloTowers).end()) 
	  info.towers.push_back(*tower);
	else 
	  LogTrace("TrackAssociator::fillEcal") << "CaloTower from the cone is not found for DetId: " << id.rawId();
     }
   
}

void TrackAssociator::fillDTSegments( const edm::Event& iEvent,
				      const edm::EventSetup& iSetup,
				      TrackDetMatchInfo& info,
				      const FreeTrajectoryState& trajectoryPoint,
				      const AssociatorParameters& parameters)
{
   TimerStack timers;
   timers.push("TrackAssociator::fillDTSegments");
   using namespace edm;
   TrajectoryStateOnSurface tSOSDest;

   // Get the DT Geometry
   ESHandle<DTGeometry> dtGeom;
   iSetup.get<MuonGeometryRecord>().get(dtGeom);
   muonDetIdAssociator_.setGeometry(&*theTrackingGeometry_);

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
         edm::LogWarning("MuonGeometry")<<"Failed to get detector unit\n";
         continue;
      }
      const Surface& surf = chamber->surface();

      LogTrace("fillDTSegments") <<"Will propagate to surface: "<<surf.position()<<" "<<surf.rotation()<<"\n";
      tSOSDest = ivProp_->Propagator::propagate(trajectoryPoint, surf);

      if (! tSOSDest.isValid()) {
         edm::LogWarning("PropagationProblem") << "Failed to propagate track to DTChamber\n";
         continue;
      }

      // Get the range for the corresponding LayerId
      DTRecSegment4DCollection::range  range = dtSegments->get((*detUnitIt));
      // Loop over the rechits of this DetUnit
      for (DTRecSegment4DCollection::const_iterator recseg = range.first; recseg!=range.second; recseg++){
	 MuonChamberMatch matchedChamber;
	 matchedChamber.id = recseg->geographicalId();
	 matchedChamber.tState = tSOSDest;
	 addMuonSegmentMatch(matchedChamber, &(*recseg), parameters);
	 info.chambers.push_back(matchedChamber);
      }
   }
}

void TrackAssociator::fillCSCSegments( const edm::Event& iEvent,
				      const edm::EventSetup& iSetup,
				      TrackDetMatchInfo& info,
				      const FreeTrajectoryState& trajectoryPoint,
				      const AssociatorParameters& parameters)
{
   TimerStack timers;
   timers.push("TrackAssociator::fillCSCSegments");
   using namespace edm;
   TrajectoryStateOnSurface tSOSDest;

   // Get the CSC Geometry
   ESHandle<CSCGeometry> cscGeom;
   iSetup.get<MuonGeometryRecord>().get(cscGeom);
   muonDetIdAssociator_.setGeometry(&*theTrackingGeometry_);

   // Get the rechit collection from the event
   timers.push("TrackAssociator::fillCSCSegments::access");
   Handle<CSCSegmentCollection> cscSegments;
   if (CSCSegmentCollectionLabels.empty())
      // iEvent_->getByType (cscSegments);
      throw cms::Exception("FatalError") << "Module lable is not set for CSCSegmentCollection.\n";
   else
      iEvent.getByLabel (CSCSegmentCollectionLabels[0], CSCSegmentCollectionLabels[1], cscSegments);
   if (! cscSegments.isValid()) 
      throw cms::Exception("FatalError") << "Unable to find CSCSegmentCollection in event!\n";

   // Iterate over all detunits
   CSCSegmentCollection::id_iterator detUnitIt;
   for (detUnitIt = cscSegments->id_begin(); detUnitIt != cscSegments->id_end(); ++detUnitIt){
      // Get the GeomDet from the setup
      const CSCChamber* chamber = cscGeom->chamber(*detUnitIt);
      if (chamber == 0){
	 edm::LogWarning("MuonGeometry")<<"Failed to get detector unit\n";
         continue;
      }
      const Surface& surf = chamber->surface();
      
      LogTrace("fillDTSegments") <<"Will propagate to surface: "<<surf.position()<<" "<<surf.rotation()<<"\n";
      tSOSDest = ivProp_->Propagator::propagate(trajectoryPoint, surf);

      if (! tSOSDest.isValid()) {
	 edm::LogWarning("PropagationProblem") << "Failed to propagate track to DTChamber\n";
         continue;
      }

      // Get the range for the corresponding LayerId
      CSCSegmentCollection::range  range = cscSegments->get((*detUnitIt));
      // Loop over the rechits of this DetUnit
      for (CSCSegmentCollection::const_iterator recseg = range.first; recseg!=range.second; recseg++){
	 MuonChamberMatch matchedChamber;
	 matchedChamber.id = recseg->geographicalId();
	 matchedChamber.tState = tSOSDest;
	 addMuonSegmentMatch(matchedChamber, &(*recseg), parameters);
	 info.chambers.push_back(matchedChamber);
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

   HepPDT::ParticleID id(track.type());
   int charge = id.threeCharge() < 0 ? -1 : 1;

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

void TrackAssociator::fillMuonSegments( const edm::Event& iEvent,
					const edm::EventSetup& iSetup,
					TrackDetMatchInfo& info,
					FreeTrajectoryState& trajectoryPoint,
					const AssociatorParameters& parameters)
{
   TimerStack timers;
   timers.push("TrackAssociator::fillMuonSegments");
   using namespace edm;

   muonDetIdAssociator_.setGeometry(&*theTrackingGeometry_);

   // Get the segments from the event
   timers.push("TrackAssociator::fillMuonSegments::access");
   Handle<DTRecSegment4DCollection> dtSegments;
   if (DTRecSegment4DCollectionLabels.empty())
     throw cms::Exception("FatalError") << "Module lable is not set for DTRecSegment4DCollection.\n";
   else
     iEvent.getByLabel (DTRecSegment4DCollectionLabels[0], DTRecSegment4DCollectionLabels[1], dtSegments);
   if (! dtSegments.isValid()) 
     throw cms::Exception("FatalError") << "Unable to find DTRecSegment4DCollection in event!\n";
   
   Handle<CSCSegmentCollection> cscSegments;
   if (CSCSegmentCollectionLabels.empty())
     throw cms::Exception("FatalError") << "Module lable is not set for CSCSegmentCollection.\n";
   else
     iEvent.getByLabel (CSCSegmentCollectionLabels[0], CSCSegmentCollectionLabels[1], cscSegments);
   if (! cscSegments.isValid()) 
     throw cms::Exception("FatalError") << "Unable to find CSCSegmentCollection in event!\n";

   ///// get a set of DetId's in a given direction
   // get the direction first
   GlobalVector direction = trajectoryPoint.momentum().unit();
   // and find chamber DetIds
   LogTrace("TrackAssociator::fillMuonSegments") << "muon direction: " << direction << "\n\t and corresponding point: " <<
     trajectoryPoint.position() <<"\n";
   
   // check the map of available segments
   // if there is no segments in a given direction at all,
   // then there is no point to fly there.
   // 
   // MISSING
   // Possible solution: quick search for presence of segments 
   // for the set of DetIds

   timers.pop_and_push("TrackAssociator::fillMuonSegments::matchChembers");
   
   // propagate track through the muon detector
   // and get a set of matches corresponding to muon chambers
   std::vector<MuonChamberMatch> matchedChambers = 
     muonDetIdAssociator_.getTrajectoryInMuonDetector(trajectoryPoint, 
						      parameters.dRMuonPreselection, 
						      parameters.muonMaxDistanceX, 
						      parameters.muonMaxDistanceY);
   LogTrace("fillMuonSegments") << "Chambers matched: " << matchedChambers.size() << "\n";
   // Iterate over all chamber matches and fill segment matching 
   // info if it's available
   timers.pop_and_push("TrackAssociator::fillMuonSegments::findSemgents");
   for(std::vector<MuonChamberMatch>::iterator matchedChamber = matchedChambers.begin(); 
       matchedChamber != matchedChambers.end(); matchedChamber++)
     {
	const GeomDet* geomDet = muonDetIdAssociator_.getGeomDet((*matchedChamber).id);
	// DT chamber
	if(const DTChamber* chamber = dynamic_cast<const DTChamber*>(geomDet) ) {
	   // Get the range for the corresponding segments
	   DTRecSegment4DCollection::range  range = dtSegments->get(chamber->id());
	   // Loop over the segments of this chamber
	   for (DTRecSegment4DCollection::const_iterator segment = range.first; segment!=range.second; segment++)
	     addMuonSegmentMatch(*matchedChamber, &(*segment), parameters);
	}else{
	   // CSC Chamber
	   if(const CSCChamber* chamber = dynamic_cast<const CSCChamber*>(geomDet) ) {
	      // Get the range for the corresponding segments
	      CSCSegmentCollection::range  range = cscSegments->get(chamber->id());
	      // Loop over the segments
	      for (CSCSegmentCollection::const_iterator segment = range.first; segment!=range.second; segment++)
		 addMuonSegmentMatch(*matchedChamber, &(*segment), parameters);
	   }else{
	      throw cms::Exception("FatalError") << "Failed to cast GeomDet object to either DTChamber or CSCChamber. Who is this guy anyway?\n";
	   }
	}
	info.chambers.push_back(*matchedChamber);
     }
}


void TrackAssociator::addMuonSegmentMatch(MuonChamberMatch& matchedChamber,
					  const RecSegment* segment,
					  const AssociatorParameters& parameters)
{
   LogTrace("TrackAssociator::addMuonSegment")
     << "Segment local position: " << segment->localPosition() << "\n"
     << std::hex << segment->geographicalId().rawId() << "\n";
   
   const GeomDet* chamber = muonDetIdAssociator_.getGeomDet(matchedChamber.id);
   TrajectoryStateOnSurface trajectoryStateOnSurface = matchedChamber.tState;
   GlobalPoint segmentGlobalPosition = chamber->toGlobal(segment->localPosition());

   LogTrace("TrackAssociator::addMuonSegment")
     << "Segment global position: " << segmentGlobalPosition << " \t (R_xy,eta,phi): "
     << segmentGlobalPosition.perp() << "," << segmentGlobalPosition.eta() << "," << segmentGlobalPosition.phi() << "\n";

   LogTrace("TrackAssociator::fillMuonSegments")
     << "\teta hit: " << segmentGlobalPosition.eta() << " \tpropagator: " << trajectoryStateOnSurface.freeState()->position().eta() << "\n"
     << "\tphi hit: " << segmentGlobalPosition.phi() << " \tpropagator: " << trajectoryStateOnSurface.freeState()->position().phi() << std::endl;

   bool isGood = false;
   bool isDTOuterStation = false;
   if( const DTChamber* chamberDT = dynamic_cast<const DTChamber*>(chamber))
     if (chamberDT->id().station()==4)
       isDTOuterStation = true;
   if( isDTOuterStation )
     {
	isGood = fabs(segmentGlobalPosition.phi()-trajectoryStateOnSurface.freeState()->position().phi()) < parameters.dRMuon;
	// Be in chamber
	isGood &= fabs(segmentGlobalPosition.eta()-trajectoryStateOnSurface.freeState()->position().eta()) < .3;
     } else isGood = sqrt( pow(segmentGlobalPosition.eta()-trajectoryStateOnSurface.freeState()->position().eta(),2) + 
			   pow(segmentGlobalPosition.phi()-trajectoryStateOnSurface.freeState()->position().phi(),2)) < parameters.dRMuon;

   if(isGood) {
      MuonSegmentMatch muonSegment;
      muonSegment.segmentGlobalPosition = getPoint(segmentGlobalPosition);
      muonSegment.segmentLocalPosition = getPoint( segment->localPosition() );
      muonSegment.segmentLocalDirection = getVector( segment->localDirection() );
      muonSegment.segmentLocalErrorXX = segment->localPositionError().xx();
      muonSegment.segmentLocalErrorYY = segment->localPositionError().yy();
      muonSegment.segmentLocalErrorXY = segment->localPositionError().xy();
      muonSegment.segmentLocalErrorDxDz = segment->localDirectionError().xx();
      muonSegment.segmentLocalErrorDyDz = segment->localDirectionError().yy();
      
      // DANGEROUS - compiler cannot guaranty parameters ordering
      AlgebraicSymMatrix segmentCovMatrix = segment->parametersError();
      muonSegment.segmentLocalErrorXDxDz = segmentCovMatrix[2][0];
      muonSegment.segmentLocalErrorYDyDz = segmentCovMatrix[3][1];

      // muon.segmentPosition_.push_back(getPoint(segmentGlobalPosition));
      muonSegment.trajectoryGlobalPosition = getPoint(trajectoryStateOnSurface.freeState()->position()) ;
      muonSegment.trajectoryLocalPosition = getPoint(chamber->toLocal(trajectoryStateOnSurface.freeState()->position()));
      muonSegment.trajectoryLocalDirection = getVector(chamber->toLocal(trajectoryStateOnSurface.freeState()->momentum().unit()));
      // muon.trajectoryDirection_.push_back(getVector(trajectoryStateOnSurface.freeState()->momentum()));
      float errXX(-1.), errYY(-1.), errXY(-1.);
      float err_dXdZ(-1.), err_dYdZ(-1.);
      float err_XdXdZ(-1.), err_YdYdZ(-1.);
      if (trajectoryStateOnSurface.freeState()->hasError()){
	 LocalError err = trajectoryStateOnSurface.localError().positionError();
	 errXX = err.xx();
	 errXY = err.xy();
	 errYY = err.yy();

	 // DANGEROUS - compiler cannot guaranty parameters ordering
	 AlgebraicSymMatrix trajectoryCovMatrix = trajectoryStateOnSurface.localError().matrix();
	 err_dXdZ = trajectoryCovMatrix[1][1];
	 err_dYdZ = trajectoryCovMatrix[2][2];
	 err_XdXdZ = trajectoryCovMatrix[3][1];
	 err_YdYdZ = trajectoryCovMatrix[4][2];
      }
      muonSegment.trajectoryLocalErrorXX = errXX;
      muonSegment.trajectoryLocalErrorXY = errXY;
      muonSegment.trajectoryLocalErrorYY = errYY;
      muonSegment.trajectoryLocalErrorDxDz = err_dXdZ;
      muonSegment.trajectoryLocalErrorDyDz = err_dYdZ;
      muonSegment.trajectoryLocalErrorXDxDz = err_XdXdZ;
      muonSegment.trajectoryLocalErrorYDyDz = err_YdYdZ;
      matchedChamber.segments.push_back(muonSegment);
   }
}
	
