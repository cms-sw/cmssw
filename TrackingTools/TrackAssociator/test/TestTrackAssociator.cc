// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      TestTrackAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: TestTrackAssociator.cc,v 1.22.2.1 2009/07/01 10:04:17 dmytro Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

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

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include <boost/regex.hpp>

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "Utilities/Timing/interface/TimerStack.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

class TestTrackAssociator : public edm::EDAnalyzer {
 public:
   explicit TestTrackAssociator(const edm::ParameterSet&);
   virtual ~TestTrackAssociator(){
      TimingReport::current()->dump(std::cout);
   }
   
   virtual void analyze (const edm::Event&, const edm::EventSetup&);

 private:
   TrackDetectorAssociator trackAssociator_;
   TrackAssociatorParameters parameters_;
};

TestTrackAssociator::TestTrackAssociator(const edm::ParameterSet& iConfig)
{
   // TrackAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( parameters );
   
   trackAssociator_.useDefaultPropagator();
}

void TestTrackAssociator::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // get list of tracks and their vertices
   Handle<reco::MuonCollection> muons;
   iEvent.getByLabel("muons",muons);
      
   // loop 
   LogVerbatim("TrackAssociator") << "Number of muons found in the event: " << muons->size() ;
   for(reco::MuonCollection::const_iterator muon = muons->begin(); 
       muon != muons->end(); ++muon){
      
      // skip low Pt tracks
      if (muon->pt() < 2) {
	 LogVerbatim("TrackAssociator") << "Skipped low Pt muon (Pt: " << muon->pt() << ")" ;
	 continue;
      }
      
      // skip tracks originated away from the IP
      if (fabs(muon->vertex().rho()) > 50) {
	 LogVerbatim("TrackAssociator") << "Skipped track with large impact parameter: " <<muon->vertex().rho();
	 continue;
      }
      
      LogVerbatim("TrackAssociator") << "\n-------------------------------------------------------\n Track (pt,eta,phi): " 
	<< muon->pt() << " , " << muon->eta() << " , " << muon->phi() ;
      
      TrackDetMatchInfo info;
      if (muon->innerTrack().isAvailable()) 
	info = trackAssociator_.associate(iEvent, iSetup, *muon->innerTrack(), parameters_);
      else {
	 if (!muon->outerTrack().isAvailable()) {
	    LogVerbatim("TrackAssociator") << "No refernced tracks are available, skim the muon";
	    continue;
	 }
	 info = trackAssociator_.associate(iEvent, iSetup, *muon->outerTrack(), parameters_);
      }
	   
      LogVerbatim("TrackAssociator") << "===========================================================================" ;
      LogVerbatim("TrackAssociator") << "ECAL RecHit energy: crossed, 3x3(max), 5x5(max), 3x3(direction), 5x5(direction), cone R0.5, generator";
      DetId centerId = info.findMaxDeposition(TrackDetMatchInfo::EcalRecHits);
      LogVerbatim("TrackAssociator") << "     " << 
	info.crossedEnergy(TrackDetMatchInfo::EcalRecHits) << ", \t" <<
	info.nXnEnergy(centerId, TrackDetMatchInfo::EcalRecHits, 1) << ", \t" <<
	info.nXnEnergy(centerId, TrackDetMatchInfo::EcalRecHits, 2) << ", \t" <<
	info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1) << ", \t" <<
	info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 2) << ", \t" <<
	info.coneEnergy(0.5, TrackDetMatchInfo::EcalRecHits) << ", \t" <<
	info.ecalTrueEnergy;
      LogVerbatim("TrackAssociator") << "ECAL trajectory point (z,Rho,eta,phi), max deposit DetId";
      LogVerbatim("TrackAssociator") << "     " <<
	"(" << info.trkGlobPosAtEcal.z() << ", " << info.trkGlobPosAtEcal.Rho() << ", " <<
	info.trkGlobPosAtEcal.eta() << ", " << info.trkGlobPosAtEcal.phi() << "), " << centerId.rawId();
      LogVerbatim("TrackAssociator") << "ECAL crossed DetIds with associated hits: (id, energy, z, perp, eta, phi)";
      for(std::vector<const EcalRecHit*>::const_iterator hit = info.crossedEcalRecHits.begin(); 
	  hit != info.crossedEcalRecHits.end(); ++hit)
	{
	   GlobalPoint point = info.getPosition((*hit)->detid());
	   LogVerbatim("TrackAssociator") << "\t" << (*hit)->detid().rawId() << ", " << (*hit)->energy() << 
	     " \t(" << point.z() << ", \t" << point.perp() << ", \t" << point.eta() << ", \t" << point.phi() << ")";
	}
      LogVerbatim("TrackAssociator") << "ECAL crossed DetIds: (id, z, perp, eta, phi)";
      for(std::vector<DetId>::const_iterator id = info.crossedEcalIds.begin(); 
	  id != info.crossedEcalIds.end(); ++id)
	{
	   GlobalPoint point = info.getPosition(*id);
	   LogVerbatim("TrackAssociator") << "\t" << id->rawId() << 
	     " \t(" << point.z() << ", \t" << point.perp() << ", \t" << point.eta() << ", \t" << point.phi() << ")";
	}
      LogVerbatim("TrackAssociator") << "ECAL associated DetIds: (id, energy, z, perp, eta, phi)";
      for(std::vector<const EcalRecHit*>::const_iterator hit = info.ecalRecHits.begin(); 
	  hit != info.ecalRecHits.end(); ++hit)
	{
	   GlobalPoint point = info.getPosition((*hit)->detid());
	   LogVerbatim("TrackAssociator") << "\t" << (*hit)->detid().rawId() << ", " << (*hit)->energy() << 
	     " \t(" << point.z() << ", \t" << point.perp() << ", \t" << point.eta() << ", \t" << point.phi() << ")";
	}
      LogVerbatim("TrackAssociator") << "Preshower crossed DetIds: (id, z, perp, eta, phi)";
      for(std::vector<DetId>::const_iterator id = info.crossedPreshowerIds.begin(); 
	  id != info.crossedPreshowerIds.end(); ++id)
	{
	   GlobalPoint point = info.getPosition(*id);
	   LogVerbatim("TrackAssociator") << "\t" << id->rawId() << 
	     " \t(" << point.z() << ", \t" << point.perp() << ", \t" << point.eta() << ", \t" << point.phi() << ")";
	}

      LogVerbatim("TrackAssociator") << "---------------------------------------------------------------------------" ;
      LogVerbatim("TrackAssociator") << "HCAL RecHit energy: crossed, 3x3(max), 5x5(max), 3x3(direction), 5x5(direction), cone R0.5, generator";
      centerId = info.findMaxDeposition(TrackDetMatchInfo::HcalRecHits);
      LogVerbatim("TrackAssociator") << "     " << 
	info.crossedEnergy(TrackDetMatchInfo::HcalRecHits) << ", \t" <<
	info.nXnEnergy(centerId, TrackDetMatchInfo::HcalRecHits, 1) << ", \t" <<
	info.nXnEnergy(centerId, TrackDetMatchInfo::HcalRecHits, 2) << ", \t" <<
	info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1) << ", \t" <<
	info.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 2) << ", \t" <<
	info.coneEnergy(0.5, TrackDetMatchInfo::HcalRecHits) << ", \t" <<
	info.hcalTrueEnergyCorrected;
      LogVerbatim("TrackAssociator") << "HCAL trajectory point (z,Rho,eta,phi), max deposit DetId";
      LogVerbatim("TrackAssociator") << "     " <<
	"(" << info.trkGlobPosAtHcal.z() << ", " << info.trkGlobPosAtHcal.Rho() << ", " <<
	info.trkGlobPosAtHcal.eta() << ", " << info.trkGlobPosAtHcal.phi() << "), " << centerId.rawId();
      LogVerbatim("TrackAssociator") << "HCAL crossed DetIds with hits:";
      for(std::vector<const HBHERecHit*>::const_iterator hit = info.crossedHcalRecHits.begin(); 
	  hit != info.crossedHcalRecHits.end(); ++hit)
	{
	   GlobalPoint point = info.getPosition((*hit)->detid());
	   LogVerbatim("TrackAssociator") << "\t" << (*hit)->detid().rawId() << ", " << (*hit)->energy() << 
	     " \t(" << point.z() << ", \t" << point.perp() << ", \t" << (*hit)->id().depth() << ", \t" << 
	     point.eta() << ", \t" << point.phi() << ")";
	}
      LogVerbatim("TrackAssociator") << "HCAL crossed DetIds: (id, z, perp, eta, phi)";
      for(std::vector<DetId>::const_iterator id = info.crossedHcalIds.begin(); 
	  id != info.crossedHcalIds.end(); ++id)
	{
	   GlobalPoint point = info.getPosition(*id);
	   LogVerbatim("TrackAssociator") << "\t" << id->rawId() << 
	     " \t(" << point.z() << ", \t" << point.perp() << ", \t" << point.eta() << ", \t" << point.phi() << ")";
	}
      LogVerbatim("TrackAssociator") << "HCAL associated DetIds: id, (energy, z, perp, depth, eta, phi)";
      for(std::vector<const HBHERecHit*>::const_iterator hit = info.hcalRecHits.begin(); 
	  hit != info.hcalRecHits.end(); ++hit)
	{
	   GlobalPoint point = info.getPosition((*hit)->detid());
	   LogVerbatim("TrackAssociator") << "\t" << (*hit)->detid().rawId() << ", " << (*hit)->energy() << 
	     " \t(" << point.z() << ", \t" << point.perp() << ", \t" << (*hit)->id().depth() << ", \t" << 
	     point.eta() << ", \t" << point.phi() << ")";
	}

      LogVerbatim("TrackAssociator") << "---------------------------------------------------------------------------" ;
      LogVerbatim("TrackAssociator") << "HO RecHit energy: crossed, 3x3(max), 5x5(max), 3x3(direction), 5x5(direction), cone R0.5";
      centerId = info.findMaxDeposition(TrackDetMatchInfo::HORecHits);
      LogVerbatim("TrackAssociator") << "     " << 
	info.crossedEnergy(TrackDetMatchInfo::HORecHits) << ", \t" <<
	info.nXnEnergy(centerId, TrackDetMatchInfo::HORecHits, 1) << ", \t" <<
	info.nXnEnergy(centerId, TrackDetMatchInfo::HORecHits, 2) << ", \t" <<
	info.nXnEnergy(TrackDetMatchInfo::HORecHits, 1) << ", \t" <<
	info.nXnEnergy(TrackDetMatchInfo::HORecHits, 2) << ", \t" <<
	info.coneEnergy(0.5, TrackDetMatchInfo::HORecHits);
      LogVerbatim("TrackAssociator") << "HO trajectory point (z,Rho,eta,phi), max deposit DetId";
      LogVerbatim("TrackAssociator") << "     " <<
	"(" << info.trkGlobPosAtHO.z() << ", " << info.trkGlobPosAtHO.Rho() << ", " <<
	info.trkGlobPosAtHO.eta() << ", " << info.trkGlobPosAtHO.phi() << "), " << centerId.rawId();
      LogVerbatim("TrackAssociator") << "HO crossed DetIds with hits:";
      for(std::vector<const HORecHit*>::const_iterator hit = info.crossedHORecHits.begin(); 
	  hit != info.crossedHORecHits.end(); ++hit)
	{
	   GlobalPoint point = info.getPosition((*hit)->detid());
	   LogVerbatim("TrackAssociator") << "\t" << (*hit)->detid().rawId() << ", " << (*hit)->energy() << 
	     " \t(" << point.z() << ", \t" << point.perp() << ", \t" << point.eta() << ", \t" << point.phi() << ")";
	}
      LogVerbatim("TrackAssociator") << "HO crossed DetIds: (id, z, perp, eta, phi)";
      for(std::vector<DetId>::const_iterator id = info.crossedHOIds.begin(); 
	  id != info.crossedHOIds.end(); ++id)
	{
	   GlobalPoint point = info.getPosition(*id);
	   LogVerbatim("TrackAssociator") << "\t" << id->rawId() << 
	     " \t(" << point.z() << ", \t" << point.perp() << ", \t" << point.eta() << ", \t" << point.phi() << ")";
	}
      LogVerbatim("TrackAssociator") << "HO associated DetIds: (id, energy,position)";
      for(std::vector<const HORecHit*>::const_iterator hit = info.hoRecHits.begin(); 
	  hit != info.hoRecHits.end(); ++hit)
	{
	   GlobalPoint point = info.getPosition((*hit)->detid());
	   LogVerbatim("TrackAssociator") << "\t" << (*hit)->detid().rawId() << ", " << (*hit)->energy() << 
	     " \t(" << point.z() << ", \t" << point.perp() << ", \t" << point.eta() << ", \t" << point.phi();
	       // << ") ## " << info.dumpGeometry((*hit)->detid());
	}

      if (parameters_.useMuon) {
	 LogVerbatim("TrackAssociator") << "Muon detector matching details: " ;
	 for(std::vector<TAMuonChamberMatch>::const_iterator chamber = info.chambers.begin();
	     chamber!=info.chambers.end(); chamber++)
	   {
	      LogVerbatim("TrackAssociator") << chamber->info() << "\n\t(DetId, station, edgeX, edgeY): "
		<< chamber->id.rawId() << ", "
		<< chamber->station() << ", "
		<< chamber->localDistanceX << ", "
		<< chamber->localDistanceY << ", ";
	      LogVerbatim("TrackAssociator") << "\t trajectory global point (z,perp,eta,phi): "
		<< chamber->tState.globalPosition().z() << ", "
		<< chamber->tState.globalPosition().perp() << ", "
		<< chamber->tState.globalPosition().eta() << ", "
		<< chamber->tState.globalPosition().phi() ;
	      LogVerbatim("TrackAssociator") << "\t trajectory local point (x,y): "
		<< chamber->tState.localPosition().x() << ", "
		<< chamber->tState.localPosition().y();

	      for(std::vector<TAMuonSegmentMatch>::const_iterator segment=chamber->segments.begin(); 
		  segment!=chamber->segments.end(); segment++)
		{
		   LogVerbatim("TrackAssociator") << "\t segment position (z,Rho,eta,phi,DetId): " 
		     << segment->segmentGlobalPosition.z() << ", "
		     << segment->segmentGlobalPosition.Rho() << ", "
		     << segment->segmentGlobalPosition.eta() << ", "
		     << segment->segmentGlobalPosition.phi() << ", "
		     << chamber->id.rawId();
		   LogVerbatim("TrackAssociator") << "\t segment local position (x,y): "
		     << segment->segmentLocalPosition.x() << ", "
		     << segment->segmentLocalPosition.y();
		}
	   }
      }
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestTrackAssociator);
