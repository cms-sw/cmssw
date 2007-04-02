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
// $Id: CaloMatchingExample.cc,v 1.4 2007/03/20 06:54:47 dmytro Exp $
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

#include "CLHEP/Random/Random.h"

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
   float ecal3x3EnergyMax_[1000];
   float ecal5x5EnergyMax_[1000];
   float ecalTrueEnergy_[1000];
   float trkPosAtEcal_[1000][2];
   float ecalMaxPos_[1000][2];

   float ecalCrossedEnergyRandom_[1000];
   float ecal3x3EnergyRandom_[1000];
   float ecal5x5EnergyRandom_[1000];
   float ecal3x3EnergyMaxRandom_[1000];
   float ecal5x5EnergyMaxRandom_[1000];
   float trkPosAtEcalRandom_[1000][2];
   float ecalMaxPosRandom_[1000][2];

   float hcalCrossedEnergy_[1000];
   float hcal3x3Energy_[1000];
   float hcal5x5Energy_[1000];
   float hcal3x3EnergyMax_[1000];
   float hcal5x5EnergyMax_[1000];
   float hcalTrueEnergy_[1000];
   float hcalTrueEnergyCorrected_[1000];
   float trkPosAtHcal_[1000][2];
   float hcalMaxPos_[1000][2];
   float trackPt_[1000];

   float hcalCrossedEnergyRandom_[1000];
   float hcal3x3EnergyRandom_[1000];
   float hcal5x5EnergyRandom_[1000];
   float hcal3x3EnergyMaxRandom_[1000];
   float hcal5x5EnergyMaxRandom_[1000];
   float trkPosAtHcalRandom_[1000][2];
   float hcalMaxPosRandom_[1000][2];

   edm::InputTag inputRecoTrackColl_;
   TrackAssociatorParameters parameters_;
};

CaloMatchingExample::CaloMatchingExample(const edm::ParameterSet& iConfig)
{
   HepRandom::createInstance();
   file_ = new TFile( iConfig.getParameter<std::string>("outputfile").c_str(), "recreate");
   tree_ = new TTree( "calomatch","calomatch" );
   tree_->Branch("nTracks",&nTracks_,"nTracks/I");
   
   tree_->Branch("ecalCrossedEnergy", ecalCrossedEnergy_, "ecalCrossedEnergy[nTracks]/F");
   tree_->Branch("ecal3x3Energy", ecal3x3Energy_, "ecal3x3Energy[nTracks]/F");
   tree_->Branch("ecal5x5Energy", ecal5x5Energy_, "ecal5x5Energy[nTracks]/F");
   tree_->Branch("ecal3x3EnergyMax", ecal3x3EnergyMax_, "ecal3x3EnergyMax[nTracks]/F");
   tree_->Branch("ecal5x5EnergyMax", ecal5x5EnergyMax_, "ecal5x5EnergyMax[nTracks]/F");
   tree_->Branch("ecalTrueEnergy", ecalTrueEnergy_, "ecalTrueEnergy[nTracks]/F");
   tree_->Branch("trkPosAtEcal", trkPosAtEcal_, "trkPosAtEcal[nTracks][2]/F");
   tree_->Branch("ecalMaxPos", ecalMaxPos_, "ecalMaxPos[nTracks][2]/F");
   
   tree_->Branch("ecalCrossedEnergyRandom", ecalCrossedEnergyRandom_, "ecalCrossedEnergyRandom[nTracks]/F");
   tree_->Branch("ecal3x3EnergyRandom", ecal3x3EnergyRandom_, "ecal3x3EnergyRandom[nTracks]/F");
   tree_->Branch("ecal5x5EnergyRandom", ecal5x5EnergyRandom_, "ecal5x5EnergyRandom[nTracks]/F");
   tree_->Branch("ecal3x3EnergyMaxRandom", ecal3x3EnergyMaxRandom_, "ecal3x3EnergyMaxRandom[nTracks]/F");
   tree_->Branch("ecal5x5EnergyMaxRandom", ecal5x5EnergyMaxRandom_, "ecal5x5EnergyMaxRandom[nTracks]/F");
   tree_->Branch("trkPosAtEcalRandom", trkPosAtEcalRandom_, "trkPosAtEcalRandom[nTracks][2]/F");
   tree_->Branch("ecalMaxPosRandom", ecalMaxPosRandom_, "ecalMaxPosRandom[nTracks][2]/F");

   tree_->Branch("hcalCrossedEnergy", hcalCrossedEnergy_, "hcalCrossedEnergy[nTracks]/F");
   tree_->Branch("hcal3x3Energy", hcal3x3Energy_, "hcal3x3Energy[nTracks]/F");
   tree_->Branch("hcal5x5Energy", hcal5x5Energy_, "hcal5x5Energy[nTracks]/F");
   tree_->Branch("hcal3x3EnergyMax", hcal3x3EnergyMax_, "hcal3x3EnergyMax[nTracks]/F");
   tree_->Branch("hcal5x5EnergyMax", hcal5x5EnergyMax_, "hcal5x5EnergyMax[nTracks]/F");
   tree_->Branch("hcalTrueEnergy", hcalTrueEnergy_, "hcalTrueEnergy[nTracks]/F");
   tree_->Branch("hcalTrueEnergyCorrected", hcalTrueEnergyCorrected_, "hcalTrueEnergyCorrected[nTracks]/F");
   tree_->Branch("trkPosAtHcal", trkPosAtHcal_, "trkPosAtHcal[nTracks][2]/F");
   tree_->Branch("hcalMaxPos", hcalMaxPos_, "hcalMaxPos[nTracks][2]/F");

   tree_->Branch("hcalCrossedEnergyRandom", hcalCrossedEnergyRandom_, "hcalCrossedEnergyRandom[nTracks]/F");
   tree_->Branch("hcal3x3EnergyRandom", hcal3x3EnergyRandom_, "hcal3x3EnergyRandom[nTracks]/F");
   tree_->Branch("hcal5x5EnergyRandom", hcal5x5EnergyRandom_, "hcal5x5EnergyRandom[nTracks]/F");
   tree_->Branch("hcal3x3EnergyMaxRandom", hcal3x3EnergyMaxRandom_, "hcal3x3EnergyMaxRandom[nTracks]/F");
   tree_->Branch("hcal5x5EnergyMaxRandom", hcal5x5EnergyMaxRandom_, "hcal5x5EnergyMaxRandom[nTracks]/F");
   tree_->Branch("trkPosAtHcalRandom", trkPosAtHcalRandom_, "trkPosAtHcalRandom[nTracks][2]/F");
   tree_->Branch("hcalMaxPosRandom", hcalMaxPosRandom_, "hcalMaxPosRandom[nTracks][2]/F");

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
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, *recoTrack, parameters_);
      // get some noise info (random direction)
      ROOT::Math::RhoEtaPhiVector randomVector(10,(HepRandom::getTheEngine()->flat()-0.5)*6,(HepRandom::getTheEngine()->flat()-0.5)*2*3.1416);
      TrackDetMatchInfo infoRandom = trackAssociator_.associate(iEvent, iSetup,      
								GlobalVector(randomVector.x(),randomVector.y(),randomVector.z()),
								GlobalPoint(0,0,0),
								+1, parameters_);
      
      ///////////////////////////////////////////////////
      //
      //   Fill ntuple
      //
      ///////////////////////////////////////////////////
      
      DetId centerId;
      DetId centerIdRandom;
      
      trackPt_[nTracks_] = recoTrack->pt();
      ecalCrossedEnergy_[nTracks_] = info.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
      centerId                     = info.findMaxDeposition(TrackDetMatchInfo::EcalRecHits);
      ecal3x3Energy_[nTracks_]     = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1);
      ecal5x5Energy_[nTracks_]     = info.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 2);
      ecal3x3EnergyMax_[nTracks_]  = info.nXnEnergy(centerId, TrackDetMatchInfo::EcalRecHits, 1);
      ecal5x5EnergyMax_[nTracks_]  = info.nXnEnergy(centerId, TrackDetMatchInfo::EcalRecHits, 2);
      ecalTrueEnergy_[nTracks_]    = info.ecalTrueEnergy;
      trkPosAtEcal_[nTracks_][0]   = info.trkGlobPosAtEcal.eta();
      trkPosAtEcal_[nTracks_][1]   = info.trkGlobPosAtEcal.phi();
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
      ecalCrossedEnergyRandom_[nTracks_] = infoRandom.crossedEnergy(TrackDetMatchInfo::EcalRecHits);
      centerIdRandom                     = infoRandom.findMaxDeposition(TrackDetMatchInfo::EcalRecHits);
      ecal3x3EnergyRandom_[nTracks_]     = infoRandom.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 1);
      ecal5x5EnergyRandom_[nTracks_]     = infoRandom.nXnEnergy(TrackDetMatchInfo::EcalRecHits, 2);
      ecal3x3EnergyMaxRandom_[nTracks_]  = infoRandom.nXnEnergy(centerIdRandom, TrackDetMatchInfo::EcalRecHits, 1);
      ecal5x5EnergyMaxRandom_[nTracks_]  = infoRandom.nXnEnergy(centerIdRandom, TrackDetMatchInfo::EcalRecHits, 2);
      trkPosAtEcalRandom_[nTracks_][0]   = infoRandom.trkGlobPosAtEcal.eta();
      trkPosAtEcalRandom_[nTracks_][1]   = infoRandom.trkGlobPosAtEcal.phi();

      hcalCrossedEnergy_[nTracks_] = info.hcalEnergy();
      centerId                     = info.findMaxDeposition(TrackDetMatchInfo::HcalRecHits);
      hcal3x3Energy_[nTracks_]     = info.nXnEnergy(centerId, TrackDetMatchInfo::HcalRecHits, 1);
      hcal5x5Energy_[nTracks_]     = info.nXnEnergy(centerId, TrackDetMatchInfo::HcalRecHits, 2);
      hcalTrueEnergy_[nTracks_]    = info.hcalTrueEnergy;
      hcalTrueEnergyCorrected_[nTracks_]    = info.hcalTrueEnergyCorrected;
      trkPosAtHcal_[nTracks_][0]   = info.trkGlobPosAtHcal.eta();
      trkPosAtHcal_[nTracks_][1]   = info.trkGlobPosAtHcal.phi();
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
      hcalCrossedEnergyRandom_[nTracks_] = infoRandom.crossedEnergy(TrackDetMatchInfo::HcalRecHits);
      centerIdRandom                     = infoRandom.findMaxDeposition(TrackDetMatchInfo::HcalRecHits);
      hcal3x3EnergyRandom_[nTracks_]     = infoRandom.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 1);
      hcal5x5EnergyRandom_[nTracks_]     = infoRandom.nXnEnergy(TrackDetMatchInfo::HcalRecHits, 2);
      hcal3x3EnergyMaxRandom_[nTracks_]  = infoRandom.nXnEnergy(centerIdRandom, TrackDetMatchInfo::HcalRecHits, 1);
      hcal5x5EnergyMaxRandom_[nTracks_]  = infoRandom.nXnEnergy(centerIdRandom, TrackDetMatchInfo::HcalRecHits, 2);
      trkPosAtHcalRandom_[nTracks_][0]   = infoRandom.trkGlobPosAtHcal.eta();
      trkPosAtHcalRandom_[nTracks_][1]   = infoRandom.trkGlobPosAtHcal.phi();
      
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
      LogTrace("TrackAssociator") << "HCAL, true energy corrected: " << info.hcalTrueEnergyCorrected << " GeV" ;
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
