// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      CaloMatchingExample
// 
/*

 Description: Example shows how to access various forms of energy deposition and store them in an ntuple

*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
// $Id: CaloMatchingExample.cc,v 1.3 2007/03/09 14:08:16 dmytro Exp $
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
   float ecalTrueEnergy_[1000];
   float trkPosAtEcal_[1000][2];
   float ecalMaxPos_[1000][2];
   
   float hcalCrossedEnergy_[1000];
   float hcal3x3Energy_[1000];
   float hcal5x5Energy_[1000];
   float hcalTrueEnergy_[1000];
   float trkPosAtHcal_[1000][2];
   float hcalMaxPos_[1000][2];
   float trackPt_[1000];
   
   edm::InputTag inputRecoTrackColl_;
   TrackAssociatorParameters parameters_;
};

CaloMatchingExample::CaloMatchingExample(const edm::ParameterSet& iConfig)
{
   
   file_ = new TFile( iConfig.getParameter<std::string>("outputfile").c_str(), "recreate");
   tree_ = new TTree( "calomatch","calomatch" );
   tree_->Branch("nTracks",&nTracks_,"nTracks/I");
   
   tree_->Branch("ecalCrossedEnergy", ecalCrossedEnergy_, "ecalCrossedEnergy[nTracks]/F");
   tree_->Branch("ecal3x3Energy", ecal3x3Energy_, "ecal3x3Energy[nTracks]/F");
   tree_->Branch("ecal5x5Energy", ecal5x5Energy_, "ecal5x5Energy[nTracks]/F");
   tree_->Branch("ecalTrueEnergy", ecalTrueEnergy_, "ecalTrueEnergy[nTracks]/F");
   tree_->Branch("trkPosAtEcal", trkPosAtEcal_, "trkPosAtEcal_[nTracks][2]/F");
   tree_->Branch("ecalMaxPos", ecalMaxPos_, "ecalMaxPos_[nTracks][2]/F");

   tree_->Branch("hcalCrossedEnergy", hcalCrossedEnergy_, "hcalCrossedEnergy[nTracks]/F");
   tree_->Branch("hcal3x3Energy", hcal3x3Energy_, "hcal3x3Energy[nTracks]/F");
   tree_->Branch("hcal5x5Energy", hcal5x5Energy_, "hcal5x5Energy[nTracks]/F");
   tree_->Branch("hcalTrueEnergy", hcalTrueEnergy_, "hcalTrueEnergy[nTracks]/F");
   tree_->Branch("trkPosAtHcal", trkPosAtHcal_, "trkPosAtHcal_[nTracks][2]/F");
   tree_->Branch("hcalMaxPos", hcalMaxPos_, "hcalMaxPos_[nTracks][2]/F");

   tree_->Branch("trackPt", trackPt_, "trackPt_[nTracks]/F");
   
   inputRecoTrackColl_ = iConfig.getParameter<edm::InputTag>("inputRecoTrackColl");
   
   // TrackAssociator parameters
   edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
   parameters_.loadParameters( parameters );

   trackAssociator_.useDefaultPropagator();
   
}

void CaloMatchingExample::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // calo geometry
   edm::ESHandle<CaloGeometry> geometry;
   iSetup.get<IdealGeometryRecord>().get(geometry);
   if (! geometry.isValid()) throw cms::Exception("FatalError") << "Unable to find IdealGeometryRecord in event!\n";

   nTracks_ = 0;
   
   // get reco tracks 
   Handle<reco::TrackCollection> recoTracks;
   iEvent.getByLabel(inputRecoTrackColl_, recoTracks);
   if (! recoTracks.isValid() ) throw cms::Exception("FatalError") << "No reco tracks were found\n";

   // loop over reconstructed tracks
   LogTrace("TrackAssociator") << "Number of reco tracks found in the event: " << recoTracks->size() ;
   // for(SimTrackContainer::const_iterator tracksCI = simTracks->begin();
   //    tracksCI != simTracks->end(); tracksCI++){
   for(reco::TrackCollection::const_iterator recoTrack = recoTracks->begin();
       recoTrack != recoTracks->end(); ++recoTrack){
       
      // skip low Pt tracks
      if (recoTrack->pt() < 2) {
	 LogTrace("TrackAssociator") << "Skipped low Pt track (Pt: " << recoTrack->pt() << ")" ;
	 continue;
      }
      
      LogTrace("TrackAssociator") << "\n-------------------------------------------------------\n Track (pt,eta,phi): " << 
	recoTrack->pt() << " , " << recoTrack->eta() << " , " << recoTrack->phi() ;
      
      // Get track matching info

      LogTrace("TrackAssociator") << "===========================================================================\nDetails:\n" ;
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup,
							  trackAssociator_.getFreeTrajectoryState(iSetup, *recoTrack),
							  parameters_);
      
      ///////////////////////////////////////////////////
      //
      //   Fill ntuple
      //
      ///////////////////////////////////////////////////
      
      DetId centerId;
      
      trackPt_[nTracks_] = recoTrack->pt();
      ecalCrossedEnergy_[nTracks_] = info.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
      centerId = info.findMaxDeposition(TrackDetMatchInfo::EcalRecHits);
      ecal3x3Energy_[nTracks_] = info.nXnEnergy(centerId, TrackDetMatchInfo::EcalRecHits, 1);
      ecal5x5Energy_[nTracks_] = info.nXnEnergy(centerId, TrackDetMatchInfo::EcalRecHits, 2);
      ecalTrueEnergy_[nTracks_] = info.ecalTrueEnergy;
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
      centerId = info.findMaxDeposition(TrackDetMatchInfo::HcalRecHits);
      hcal3x3Energy_[nTracks_] = info.nXnEnergy(centerId, TrackDetMatchInfo::HcalRecHits, 1);
      hcal5x5Energy_[nTracks_] = info.nXnEnergy(centerId, TrackDetMatchInfo::HcalRecHits, 2);
      hcalTrueEnergy_[nTracks_] = info.hcalTrueEnergy;
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
      LogTrace("TrackAssociator") << "ECAL, true energy: " << info.ecalTrueEnergy << " GeV" ;
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
      LogTrace("TrackAssociator") << "HCAL, true energy: " << info.hcalTrueEnergy << " GeV" ;
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
