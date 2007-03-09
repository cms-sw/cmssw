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
// $Id: TestTrackAssociator.cc,v 1.11 2007/03/08 04:19:27 dmytro Exp $
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
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"

class TestTrackAssociator : public edm::EDAnalyzer {
 public:
   explicit TestTrackAssociator(const edm::ParameterSet&);
   virtual ~TestTrackAssociator(){
      TimingReport::current()->dump(std::cout);
   }
   
   virtual void analyze (const edm::Event&, const edm::EventSetup&);

 private:
   TrackDetectorAssociator trackAssociator_;
   bool useEcal_;
   bool useHcal_;
   bool useMuon_;
   bool useOldMuonMatching_;
};

TestTrackAssociator::TestTrackAssociator(const edm::ParameterSet& iConfig)
{
   useEcal_ = iConfig.getParameter<bool>("useEcal");
   useHcal_ = iConfig.getParameter<bool>("useHcal");
   useMuon_ = iConfig.getParameter<bool>("useMuon");
   if (iConfig.getParameter<bool>("disableTimers")){
      TimerStack timers; 
      timers.disableAllTimers();
   }
	
   useOldMuonMatching_ = iConfig.getParameter<bool>("useOldMuonMatching");
   
   // Fill data labels
   trackAssociator_.theEBRecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("EBRecHitCollectionLabel");
   trackAssociator_.theEERecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("EERecHitCollectionLabel");
   trackAssociator_.theCaloTowerCollectionLabel = iConfig.getParameter<edm::InputTag>("CaloTowerCollectionLabel");
   trackAssociator_.theHBHERecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("HBHERecHitCollectionLabel");
   trackAssociator_.theHORecHitCollectionLabel = iConfig.getParameter<edm::InputTag>("HORecHitCollectionLabel");
   trackAssociator_.theDTRecSegment4DCollectionLabel = iConfig.getParameter<edm::InputTag>("DTRecSegment4DCollectionLabel");
   trackAssociator_.theCSCSegmentCollectionLabel = iConfig.getParameter<edm::InputTag>("CSCSegmentCollectionLabel");

   trackAssociator_.useDefaultPropagator();
}

void TestTrackAssociator::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // get list of tracks and their vertices
   Handle<SimTrackContainer> simTracks;
   iEvent.getByType<SimTrackContainer>(simTracks);
   
   Handle<SimVertexContainer> simVertices;
   iEvent.getByType<SimVertexContainer>(simVertices);
   if (! simVertices.isValid() ) throw cms::Exception("FatalError") << "No vertices found\n";
   
   // loop over simulated tracks
   LogVerbatim("TrackAssociator") << "Number of simulated tracks found in the event: " << simTracks->size() ;
   for(SimTrackContainer::const_iterator tracksCI = simTracks->begin(); 
       tracksCI != simTracks->end(); tracksCI++){
      
      // skip low Pt tracks
      if (tracksCI->momentum().perp() < 5) {
	 LogVerbatim("TrackAssociator") << "Skipped low Pt track (Pt: " << tracksCI->momentum().perp() << ")" ;
	 continue;
      }
      
      // get vertex
      int vertexIndex = tracksCI->vertIndex();
      // uint trackIndex = tracksCI->genpartIndex();
      
      SimVertex vertex(Hep3Vector(0.,0.,0.),0);
      if (vertexIndex >= 0) vertex = (*simVertices)[vertexIndex];
      
      // skip tracks originated away from the IP
      if (vertex.position().rho() > 50) {
	 LogVerbatim("TrackAssociator") << "Skipped track originated away from IP: " <<vertex.position().rho();
	 continue;
      }
      
      LogVerbatim("TrackAssociator") << "\n-------------------------------------------------------\n Track (pt,eta,phi): " << tracksCI->momentum().perp() << " , " <<
	tracksCI->momentum().eta() << " , " << tracksCI->momentum().phi() ;
      
	if (1==2){ // it's just an example, and we don't need it for tests
	   // Simply get ECAL energy of the crossed crystals
	   if (useEcal_)
	     LogVerbatim("TrackAssociator") << "ECAL energy of crossed crystals: " << 
	     trackAssociator_.getEcalEnergy(iEvent, iSetup,
					    trackAssociator_.getFreeTrajectoryState(iSetup, *tracksCI, vertex) )
	       << " GeV" ;
	}
				       
      // Get HCAL energy in more generic way
      TrackDetectorAssociator::AssociatorParameters parameters;
      parameters.useEcal = useEcal_ ;
      parameters.useHcal = useHcal_ ;
      parameters.useHO = useHcal_ ;
      parameters.useCalo = useHcal_ ;
      parameters.useMuon = useMuon_ ;
      parameters.dREcal = 0.03;
      parameters.dRHcal = 0.2;
      parameters.dRMuon = 0.1;
//      parameters.dRMuonPreselection = 0.5;
      parameters.useOldMuonMatching = useOldMuonMatching_;
      
      LogVerbatim("TrackAssociator") << "===========================================================================\nDetails:\n" ;
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup,
							  trackAssociator_.getFreeTrajectoryState(iSetup, *tracksCI, vertex),
							  parameters);
      LogVerbatim("TrackAssociator") << "ECAL, number of crossed cells: " << info.crossedEcalRecHits.size() ;
      LogVerbatim("TrackAssociator") << "ECAL, energy of crossed cells: " << info.ecalEnergy() << " GeV" ;
      LogVerbatim("TrackAssociator") << "ECAL, number of cells in the cone: " << info.ecalRecHits.size() ;
      LogVerbatim("TrackAssociator") << "ECAL, energy in the cone: " << info.ecalConeEnergy() << " GeV" ;
      LogVerbatim("TrackAssociator") << "ECAL, trajectory point (z,Rho,eta,phi): " << info.trkGlobPosAtEcal.z() << ", "
	<< info.trkGlobPosAtEcal.Rho() << " , "	<< info.trkGlobPosAtEcal.eta() << " , " 
	<< info.trkGlobPosAtEcal.phi();
      
      LogVerbatim("TrackAssociator") << "HCAL, number of crossed elements (towers): " << info.crossedTowers.size() ;
      LogVerbatim("TrackAssociator") << "HCAL, energy of crossed elements (towers): " << info.hcalTowerEnergy() << " GeV" ;
      LogVerbatim("TrackAssociator") << "HCAL, number of crossed elements (hits): "   << info.crossedHcalRecHits.size() ;
      LogVerbatim("TrackAssociator") << "HCAL, energy of crossed elements (hits): "   << info.hcalEnergy() << " GeV" ;
      LogVerbatim("TrackAssociator") << "HCAL, number of elements in the cone (towers): " << info.towers.size() ;
      LogVerbatim("TrackAssociator") << "HCAL, energy in the cone (towers): "             << info.hcalTowerConeEnergy() << " GeV" ;
      LogVerbatim("TrackAssociator") << "HCAL, number of elements in the cone (hits): "   << info.hcalRecHits.size() ;
      LogVerbatim("TrackAssociator") << "HCAL, energy in the cone (hits): "               << info.hcalConeEnergy() << " GeV" ;
      LogVerbatim("TrackAssociator") << "HCAL, trajectory point (z,Rho,eta,phi): " << info.trkGlobPosAtHcal.z() << ", "
	<< info.trkGlobPosAtHcal.Rho() << " , "	<< info.trkGlobPosAtHcal.eta() << " , "
	<< info.trkGlobPosAtHcal.phi();
      
      LogVerbatim("TrackAssociator") << "HO, number of crossed elements (hits): " << info.crossedHORecHits.size() ;
      LogVerbatim("TrackAssociator") << "HO, energy of crossed elements (hits): " << info.hoEnergy() << " GeV" ;
      LogVerbatim("TrackAssociator") << "HO, number of elements in the cone: " << info.hoRecHits.size() ;
      LogVerbatim("TrackAssociator") << "HO, energy in the cone: " << info.hoConeEnergy() << " GeV" ;
      LogVerbatim("TrackAssociator") << "HCAL, trajectory point (z,Rho,eta,phi): " << info.trkGlobPosAtHO.z() << ", "
	<< info.trkGlobPosAtHO.Rho() << " , "	<< info.trkGlobPosAtHO.eta() << " , "
	<< info.trkGlobPosAtHO.phi();

      if (useMuon_) {
	 LogVerbatim("TrackAssociator") << "Muon detector matching details: " ;
	 for(std::vector<MuonChamberMatch>::const_iterator chamber = info.chambers.begin();
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

	      for(std::vector<MuonSegmentMatch>::const_iterator segment=chamber->segments.begin(); 
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
