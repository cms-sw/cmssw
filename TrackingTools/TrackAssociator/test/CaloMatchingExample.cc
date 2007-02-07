// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      CaloMatchingExample
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: CaloMatchingExample.cc,v 1.9 2007/01/30 18:40:01 dmytro Exp $
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

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TimerStack.h"

#include "TFile.h"
#include "TTree.h"

class CaloMatchingExample : public edm::EDAnalyzer {
 public:
   explicit CaloMatchingExample(const edm::ParameterSet&);
   virtual ~CaloMatchingExample(){
      TimingReport::current()->dump(std::cout);
      file_->cd();
      tree_->Write();
      file_->Close();
   }
   
   virtual void analyze (const edm::Event&, const edm::EventSetup&);

 private:
   TrackDetectorAssociator trackAssociator_;
   bool useEcal_;
   bool useHcal_;
   bool useMuon_;
   bool useOldMuonMatching_;
   TFile* file_;
   TTree* tree_;
   int nTracks_;
   
   float ecalCrossedEnergy_[1000];
   float ecal3x3Energy_[1000];
   float ecal5x5Energy_[1000];
   float trkPosAtEcal_[1000][2];
   float ecalMaxPos_[1000][2];
   
   float hcalCrossedEnergy_[1000];
   float hcal3x3Energy_[1000];
   float hcal5x5Energy_[1000];
   float trkPosAtHcal_[1000][2];
   float hcalMaxPos_[1000][2];
   float trackPt_[1000];
};

CaloMatchingExample::CaloMatchingExample(const edm::ParameterSet& iConfig)
{
   
   file_ = new TFile( iConfig.getParameter<std::string>("outputfile").c_str(), "recreate");
   tree_ = new TTree( "calomatch","calomatch" );
   tree_->Branch("nTracks",&nTracks_,"nTracks/I");
   
   tree_->Branch("ecalCrossedEnergy", ecalCrossedEnergy_, "ecalCrossedEnergy[nTracks]/F");
   tree_->Branch("ecal3x3Energy", ecal3x3Energy_, "ecal3x3Energy[nTracks]/F");
   tree_->Branch("ecal5x5Energy", ecal5x5Energy_, "ecal5x5Energy[nTracks]/F");
   tree_->Branch("trkPosAtEcal", trkPosAtEcal_, "trkPosAtEcal_[nTracks][2]/F");
   tree_->Branch("ecalMaxPos", ecalMaxPos_, "ecalMaxPos_[nTracks][2]/F");

   tree_->Branch("hcalCrossedEnergy", hcalCrossedEnergy_, "hcalCrossedEnergy[nTracks]/F");
   tree_->Branch("hcal3x3Energy", hcal3x3Energy_, "hcal3x3Energy[nTracks]/F");
   tree_->Branch("hcal5x5Energy", hcal5x5Energy_, "hcal5x5Energy[nTracks]/F");
   tree_->Branch("trkPosAtHcal", trkPosAtHcal_, "trkPosAtHcal_[nTracks][2]/F");
   tree_->Branch("hcalMaxPos", hcalMaxPos_, "hcalMaxPos_[nTracks][2]/F");

   tree_->Branch("trackPt", trackPt_, "trackPt_[nTracks]/F");
   
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

void CaloMatchingExample::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // get list of tracks and their vertices
   Handle<SimTrackContainer> simTracks;
   iEvent.getByType<SimTrackContainer>(simTracks);
   
   Handle<SimVertexContainer> simVertices;
   iEvent.getByType<SimVertexContainer>(simVertices);
   if (! simVertices.isValid() ) throw cms::Exception("FatalError") << "No vertices found\n";
   
   // calo geometry
   edm::ESHandle<CaloGeometry> geometry;
   iSetup.get<IdealGeometryRecord>().get(geometry);
   if (! geometry.isValid()) throw cms::Exception("FatalError") << "Unable to find IdealGeometryRecord in event!\n";

   nTracks_ = 0;
   
   // loop over simulated tracks
   LogTrace("TrackAssociator") << "Number of simulated tracks found in the event: " << simTracks->size() ;
   for(SimTrackContainer::const_iterator tracksCI = simTracks->begin();
       tracksCI != simTracks->end(); tracksCI++){
       
      // skip low Pt tracks
      if (tracksCI->momentum().perp() < 2) {
	 LogTrace("TrackAssociator") << "Skipped low Pt track (Pt: " << tracksCI->momentum().perp() << ")" ;
	 continue;
      }
      
      // get vertex
      int vertexIndex = tracksCI->vertIndex();
      // uint trackIndex = tracksCI->genpartIndex();
      
      SimVertex vertex(Hep3Vector(0.,0.,0.),0);
      if (vertexIndex >= 0) vertex = (*simVertices)[vertexIndex];
      
      // skip tracks originated away from the IP
      if (vertex.position().rho() > 50) {
	 LogTrace("TrackAssociator") << "Skipped track originated away from IP: " <<vertex.position().rho();
	 continue;
      }
      
      LogTrace("TrackAssociator") << "\n-------------------------------------------------------\n Track (pt,eta,phi): " << 
	tracksCI->momentum().perp() << " , " <<	tracksCI->momentum().eta() << " , " << tracksCI->momentum().phi() ;
      
      // Get track matching info
      TrackDetectorAssociator::AssociatorParameters parameters;
      parameters.useEcal = true ;
      parameters.useHcal = true ;
      parameters.useHO = true ;
      parameters.useCalo = true ;
      parameters.useMuon = false ;
      parameters.dREcalPreselection = 0.3; //should be enough for 5x5 even in EE 
      parameters.dREcal = 0.3;
      parameters.dRHcalPreselection = 1.; //should be enough for 5x5 even in HE 
      parameters.dRHcal = 1.;
      parameters.useOldMuonMatching = false;
      
      LogTrace("TrackAssociator") << "===========================================================================\nDetails:\n" ;
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup,
							  trackAssociator_.getFreeTrajectoryState(iSetup, *tracksCI, vertex),
							  parameters);
      ///////////////////////////////////////////////////
      //
      //   Fill ntuple
      //
      ///////////////////////////////////////////////////
      
      DetId centerId;
      
      trackPt_[nTracks_] = tracksCI->momentum().perp();
      ecalCrossedEnergy_[nTracks_] = info.ecalEnergy();
      centerId = info.findEcalMaxDeposition();
      ecal3x3Energy_[nTracks_] = info.ecalNxNEnergy(centerId, 1);
      ecal5x5Energy_[nTracks_] = info.ecalNxNEnergy(centerId, 2);
      trkPosAtEcal_[nTracks_][0] = info.trkGlobPosAtEcal.eta();
      trkPosAtEcal_[nTracks_][1] = info.trkGlobPosAtEcal.phi();
      if ( geometry->getSubdetectorGeometry(centerId) &&
	   geometry->getSubdetectorGeometry(centerId)->getGeometry(centerId) ) 
	{
	   GlobalPoint position = geometry->getSubdetectorGeometry(centerId)->getGeometry(centerId)->getPosition();
	   ecalMaxPos_[nTracks_][0] = position.eta();
	   ecalMaxPos_[nTracks_][1] = position.phi();
	}
      else
	{
	   ecalMaxPos_[nTracks_][0] = -999;
	   ecalMaxPos_[nTracks_][1] = -999;
	}

      hcalCrossedEnergy_[nTracks_] = info.hcalEnergy();
      centerId = info.findHcalMaxDeposition();
      hcal3x3Energy_[nTracks_] = info.hcalNxNEnergy(centerId, 1);
      hcal5x5Energy_[nTracks_] = info.hcalNxNEnergy(centerId, 2);
      trkPosAtHcal_[nTracks_][0] = info.trkGlobPosAtHcal.eta();
      trkPosAtHcal_[nTracks_][1] = info.trkGlobPosAtHcal.phi();
      if ( geometry->getSubdetectorGeometry(centerId) &&
	   geometry->getSubdetectorGeometry(centerId)->getGeometry(centerId) ) 
	{
	   GlobalPoint position = geometry->getSubdetectorGeometry(centerId)->getGeometry(centerId)->getPosition();
	   hcalMaxPos_[nTracks_][0] = position.eta();
	   hcalMaxPos_[nTracks_][1] = position.phi();
	}
      else
	{
	   hcalMaxPos_[nTracks_][0] = -999;
	   hcalMaxPos_[nTracks_][1] = -999;
	}
      
      // Debugging information 
      
      LogTrace("TrackAssociator") << "ECAL, number of crossed cells: " << info.crossedEcalRecHits.size() ;
      LogTrace("TrackAssociator") << "ECAL, energy of crossed cells: " << info.ecalEnergy() << " GeV" ;
      LogTrace("TrackAssociator") << "ECAL, number of cells in the cone: " << info.ecalRecHits.size() ;
      LogTrace("TrackAssociator") << "ECAL, energy in the cone: " << info.ecalConeEnergy() << " GeV" ;
      LogTrace("TrackAssociator") << "ECAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtEcal.z() << ", "
	<< info.trkGlobPosAtEcal.R() << " , "	<< info.trkGlobPosAtEcal.eta() << " , " 
	<< info.trkGlobPosAtEcal.phi();
      
      LogTrace("TrackAssociator") << "HCAL, number of crossed elements (towers): " << info.crossedTowers.size() ;
      LogTrace("TrackAssociator") << "HCAL, energy of crossed elements (towers): " << info.hcalTowerEnergy() << " GeV" ;
      LogTrace("TrackAssociator") << "HCAL, number of crossed elements (hits): "   << info.crossedHcalRecHits.size() ;
      LogTrace("TrackAssociator") << "HCAL, energy of crossed elements (hits): "   << info.hcalEnergy() << " GeV" ;
      LogTrace("TrackAssociator") << "HCAL, number of elements in the cone (towers): " << info.towers.size() ;
      LogTrace("TrackAssociator") << "HCAL, energy in the cone (towers): "             << info.hcalTowerConeEnergy() << " GeV" ;
      LogTrace("TrackAssociator") << "HCAL, number of elements in the cone (hits): "   << info.hcalRecHits.size() ;
      LogTrace("TrackAssociator") << "HCAL, energy in the cone (hits): "               << info.hcalConeEnergy() << " GeV" ;
      LogTrace("TrackAssociator") << "HCAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtHcal.z() << ", "
	<< info.trkGlobPosAtHcal.R() << " , "	<< info.trkGlobPosAtHcal.eta() << " , "
	<< info.trkGlobPosAtHcal.phi();
      
      LogTrace("TrackAssociator") << "HO, number of crossed elements (hits): " << info.crossedHORecHits.size() ;
      LogTrace("TrackAssociator") << "HO, energy of crossed elements (hits): " << info.hoEnergy() << " GeV" ;
      LogTrace("TrackAssociator") << "HO, number of elements in the cone: " << info.hoRecHits.size() ;
      LogTrace("TrackAssociator") << "HO, energy in the cone: " << info.hoConeEnergy() << " GeV" ;
      LogTrace("TrackAssociator") << "HCAL, trajectory point (z,R,eta,phi): " << info.trkGlobPosAtHO.z() << ", "
	<< info.trkGlobPosAtHO.R() << " , "	<< info.trkGlobPosAtHO.eta() << " , "
	<< info.trkGlobPosAtHO.phi();
       nTracks_++;
   }
   tree_->Fill();
}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloMatchingExample);
