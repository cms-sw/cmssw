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
// $Id: TestTrackAssociator.cc,v 1.5 2007/01/21 15:30:36 dmytro Exp $
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
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/OrphanHandle.h"

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

#include "Geometry/Surface/interface/Cylinder.h"
#include "Geometry/Surface/interface/Plane.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include <boost/regex.hpp>

#include "TrackingTools/TrackAssociator/interface/TrackAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"

class TestTrackAssociator : public edm::EDAnalyzer {
 public:
   explicit TestTrackAssociator(const edm::ParameterSet&);
   virtual ~TestTrackAssociator(){
      TimingReport::current()->dump(std::cout);
   }
   
   virtual void analyze (const edm::Event&, const edm::EventSetup&);

 private:
   TrackAssociator trackAssociator_;
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
   std::vector<std::string> labels = iConfig.getParameter<std::vector<std::string> >("labels");
   boost::regex regExp1 ("([^\\s,]+)[\\s,]+([^\\s,]+)$");
   boost::regex regExp2 ("([^\\s,]+)[\\s,]+([^\\s,]+)[\\s,]+([^\\s,]+)$");
   boost::smatch matches;
	

   for(std::vector<std::string>::const_iterator label = labels.begin(); label != labels.end(); label++) {
      if (boost::regex_match(*label,matches,regExp1))
	trackAssociator_.addDataLabels(matches[1],matches[2]);
      else if (boost::regex_match(*label,matches,regExp2))
	trackAssociator_.addDataLabels(matches[1],matches[2],matches[3]);
      else
	edm::LogError("ConfigurationError") << "Failed to parse label:\n" << *label << "Skipped.\n";
   }
   
   // trackAssociator_.addDataLabels("EBRecHitCollection","ecalrechit","EcalRecHitsEB");
   // trackAssociator_.addDataLabels("CaloTowerCollection","towermaker");
   // trackAssociator_.addDataLabels("DTRecSegment4DCollection","recseg4dbuilder");
   
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
   LogVerbatim("info") << "Number of simulated tracks found in the event: " << simTracks->size() ;
   for(SimTrackContainer::const_iterator tracksCI = simTracks->begin(); 
       tracksCI != simTracks->end(); tracksCI++){
      
      // skip low Pt tracks
      if (tracksCI->momentum().perp() < 5) {
	 LogVerbatim("info") << "Skipped low Pt track (Pt: " << tracksCI->momentum().perp() << ")" ;
	 continue;
      }
      
      // get vertex
      int vertexIndex = tracksCI->vertIndex();
      // uint trackIndex = tracksCI->genpartIndex();
      
      SimVertex vertex(Hep3Vector(0.,0.,0.),0);
      if (vertexIndex >= 0) vertex = (*simVertices)[vertexIndex];
      
      // skip tracks originated away from the IP
      if (vertex.position().rho() > 50) {
	 LogVerbatim("info") << "Skipped track originated away from IP: " <<vertex.position().rho();
	 continue;
      }
      
      LogVerbatim("info") << "\n-------------------------------------------------------\n Track (pt,eta,phi): " << tracksCI->momentum().perp() << " , " <<
	tracksCI->momentum().eta() << " , " << tracksCI->momentum().phi() ;
      
	if (1==2){ // it's just an example, and we don't need it for tests
	   // Simply get ECAL energy of the crossed crystals
	   if (useEcal_)
	     LogVerbatim("info") << "ECAL energy of crossed crystals: " << 
	     trackAssociator_.getEcalEnergy(iEvent, iSetup,
					    trackAssociator_.getFreeTrajectoryState(iSetup, *tracksCI, vertex) )
	       << " GeV" ;
	}
				       
      // Get HCAL energy in more generic way
      TrackAssociator::AssociatorParameters parameters;
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
      
      LogVerbatim("info") << "===========================================================================\nDetails:\n" ;
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup,
							  trackAssociator_.getFreeTrajectoryState(iSetup, *tracksCI, vertex),
							  parameters);
      LogVerbatim("info") << "ECAL, number of crossed cells: " << info.crossedEcalRecHits.size() ;
      LogVerbatim("info") << "ECAL, energy of crossed cells: " << info.ecalEnergy() << " GeV" ;
      LogVerbatim("info") << "ECAL, number of cells in the cone: " << info.ecalRecHits.size() ;
      LogVerbatim("info") << "ECAL, energy in the cone: " << info.ecalConeEnergy() << " GeV" ;
      LogVerbatim("info") << "ECAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtEcal.z() << ", "
	<< info.trkGlobPosAtEcal.R() << " , "	<< info.trkGlobPosAtEcal.eta() << " , " 
	<< info.trkGlobPosAtEcal.phi();
      
      LogVerbatim("info") << "HCAL, number of crossed elements (towers): " << info.crossedTowers.size() ;
      LogVerbatim("info") << "HCAL, energy of crossed elements (towers): " << info.hcalTowerEnergy() << " GeV" ;
      LogVerbatim("info") << "HCAL, number of crossed elements (hits): "   << info.crossedHcalRecHits.size() ;
      LogVerbatim("info") << "HCAL, energy of crossed elements (hits): "   << info.hcalEnergy() << " GeV" ;
      LogVerbatim("info") << "HCAL, number of elements in the cone (towers): " << info.towers.size() ;
      LogVerbatim("info") << "HCAL, energy in the cone (towers): "             << info.hcalTowerConeEnergy() << " GeV" ;
      LogVerbatim("info") << "HCAL, number of elements in the cone (hits): "   << info.hcalRecHits.size() ;
      LogVerbatim("info") << "HCAL, energy in the cone (hits): "               << info.hcalConeEnergy() << " GeV" ;
      LogVerbatim("info") << "HCAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtHcal.z() << ", "
	<< info.trkGlobPosAtHcal.R() << " , "	<< info.trkGlobPosAtHcal.eta() << " , "
	<< info.trkGlobPosAtHcal.phi();
      
      LogVerbatim("info") << "HO, number of crossed elements (hits): " << info.crossedHORecHits.size() ;
      LogVerbatim("info") << "HO, energy of crossed elements (hits): " << info.hoEnergy() << " GeV" ;
      LogVerbatim("info") << "HO, number of elements in the cone: " << info.hoRecHits.size() ;
      LogVerbatim("info") << "HO, energy in the cone: " << info.hoConeEnergy() << " GeV" ;
      LogVerbatim("info") << "HCAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtHO.z() << ", "
	<< info.trkGlobPosAtHO.R() << " , "	<< info.trkGlobPosAtHO.eta() << " , "
	<< info.trkGlobPosAtHO.phi();

      if (useMuon_) {
	 LogVerbatim("info") << "Muon detector matching details: " ;
	 for(std::vector<MuonChamberMatch>::const_iterator chamber = info.chambers.begin();
	     chamber!=info.chambers.end(); chamber++)
	   {
	      LogVerbatim("info") << chamber->info() << "\n\t(DetId, station, edgeX, edgeY): "
		<< chamber->id.rawId() << ", "
		<< chamber->station() << ", "
		<< chamber->localDistanceX << ", "
		<< chamber->localDistanceY << ", ";
	      for(std::vector<MuonSegmentMatch>::const_iterator segment=chamber->segments.begin(); 
		  segment!=chamber->segments.end(); segment++)
		{
		   LogVerbatim("info") << "\t trajectory global point (z,R,eta,phi): "
		     << segment->trajectoryGlobalPosition.z() << ", "
		     << segment->trajectoryGlobalPosition.R() << ", "
		     << segment->trajectoryGlobalPosition.eta() << ", "
		     << segment->trajectoryGlobalPosition.phi() ;
		   LogVerbatim("info") << "\t trajectory local point (x,y): "
		     << segment->trajectoryLocalPosition.x() << ", "
		     << segment->trajectoryLocalPosition.y();

		   LogVerbatim("info") << "\t segment position (z,R,eta,phi,DetId): " 
		     << segment->segmentGlobalPosition.z() << ", "
		     << segment->segmentGlobalPosition.R() << ", "
		     << segment->segmentGlobalPosition.eta() << ", "
		     << segment->segmentGlobalPosition.phi() << ", "
		     << chamber->id.rawId();
		   LogVerbatim("info") << "\t segment local position (x,y): "
		     << segment->segmentLocalPosition.x() << ", "
		     << segment->segmentLocalPosition.y();
		}
	   }
      }
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestTrackAssociator);
