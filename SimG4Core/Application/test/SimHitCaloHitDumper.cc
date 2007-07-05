// system include files
#include <memory>
#include <vector>

// user include files
#include "SimG4Core/Application/test/SimHitCaloHitDumper.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

void SimHitCaloHitDumper::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ){

   using namespace edm;

   std::vector<PSimHit> theTrackerHits;
   std::vector<PSimHit> theMuonHits;
   std::vector<PCaloHit> theCaloHits;

   Handle<PSimHitContainer> PixelBarrelHitsLowTof;
   Handle<PSimHitContainer> PixelBarrelHitsHighTof;
   Handle<PSimHitContainer> PixelEndcapHitsLowTof;
   Handle<PSimHitContainer> PixelEndcapHitsHighTof;
   Handle<PSimHitContainer> TIBHitsLowTof;
   Handle<PSimHitContainer> TIBHitsHighTof;
   Handle<PSimHitContainer> TIDHitsLowTof;
   Handle<PSimHitContainer> TIDHitsHighTof;
   Handle<PSimHitContainer> TOBHitsLowTof;
   Handle<PSimHitContainer> TOBHitsHighTof;
   Handle<PSimHitContainer> TECHitsLowTof;
   Handle<PSimHitContainer> TECHitsHighTof;

   Handle<PSimHitContainer> DTHits;
   Handle<PSimHitContainer> CSCHits;
   Handle<PSimHitContainer> RPCHits;

   Handle<PCaloHitContainer> EBHits;
   Handle<PCaloHitContainer> EEHits;
   Handle<PCaloHitContainer> ESHits;
   Handle<PCaloHitContainer> HcalHits;
   Handle<PCaloHitContainer> CaloTkHits;

   iEvent.getByLabel("g4SimHits","TrackerHitsPixelBarrelLowTof", PixelBarrelHitsLowTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsPixelBarrelHighTof", PixelBarrelHitsHighTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsPixelEndcapLowTof", PixelEndcapHitsLowTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsPixelEndcapHighTof", PixelEndcapHitsHighTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsTIBLowTof", TIBHitsLowTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsTIBHighTof", TIBHitsHighTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsTIDLowTof", TIDHitsLowTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsTIDHighTof", TIDHitsHighTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsTOBLowTof", TOBHitsLowTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsTOBHighTof", TOBHitsHighTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsTECLowTof", TECHitsLowTof);
   iEvent.getByLabel("g4SimHits","TrackerHitsTECHighTof", TECHitsHighTof);

   iEvent.getByLabel("g4SimHits","MuonDTHits",DTHits);
   iEvent.getByLabel("g4SimHits","MuonCSCHits",CSCHits);
   iEvent.getByLabel("g4SimHits","MuonRPCHits",RPCHits);

   iEvent.getByLabel("g4SimHits","EcalHitsEB", EBHits );
   iEvent.getByLabel("g4SimHits","EcalHitsEE", EEHits );
   iEvent.getByLabel("g4SimHits","EcalHitsES", ESHits );
   iEvent.getByLabel("g4SimHits","HcalHits", HcalHits );
   iEvent.getByLabel("g4SimHits","CaloHitsTk", CaloTkHits );

   theTrackerHits.insert(theTrackerHits.end(), PixelBarrelHitsLowTof->begin(), PixelBarrelHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), PixelBarrelHitsHighTof->begin(), PixelBarrelHitsHighTof->end());
   theTrackerHits.insert(theTrackerHits.end(), PixelEndcapHitsLowTof->begin(), PixelEndcapHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), PixelEndcapHitsHighTof->begin(), PixelEndcapHitsHighTof->end());
   theTrackerHits.insert(theTrackerHits.end(), TIBHitsLowTof->begin(), TIBHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), TIBHitsHighTof->begin(), TIBHitsHighTof->end());
   theTrackerHits.insert(theTrackerHits.end(), TIDHitsLowTof->begin(), TIDHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), TIDHitsHighTof->begin(), TIDHitsHighTof->end());
   theTrackerHits.insert(theTrackerHits.end(), TOBHitsLowTof->begin(), TOBHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), TOBHitsHighTof->begin(), TOBHitsHighTof->end());
   theTrackerHits.insert(theTrackerHits.end(), TECHitsLowTof->begin(), TECHitsLowTof->end()); 
   theTrackerHits.insert(theTrackerHits.end(), TECHitsHighTof->begin(), TECHitsHighTof->end());

   theMuonHits.insert(theMuonHits.end(), DTHits->begin(), DTHits->end() );
   theMuonHits.insert(theMuonHits.end(), CSCHits->begin(), CSCHits->end() );
   theMuonHits.insert(theMuonHits.end(), RPCHits->begin(), RPCHits->end() );

   theCaloHits.insert(theCaloHits.end(), EBHits->begin(), EBHits->end() );
   theCaloHits.insert(theCaloHits.end(), EEHits->begin(), EEHits->end() );
   theCaloHits.insert(theCaloHits.end(), ESHits->begin(), ESHits->end() );
   theCaloHits.insert(theCaloHits.end(), HcalHits->begin(), HcalHits->end() );
   theCaloHits.insert(theCaloHits.end(), CaloTkHits->begin(), CaloTkHits->end() );

   std::cout << "\n SimHit / CaloHit structure dump \n" << std::endl;
   std::cout << " Tracker Hits in the event = " << theTrackerHits.size() << std::endl; 
   std::cout << "\n" << std::endl;
   for (std::vector<PSimHit>::iterator isim = theTrackerHits.begin();
	isim != theTrackerHits.end(); ++isim){
     std::cout << (*isim) << " Track Id = " << isim->trackId() << std::endl;
   }

   std::cout << "\n Muon Hits in the event = " << theMuonHits.size() << std::endl; 
   std::cout << "\n" << std::endl;
   for (std::vector<PSimHit>::iterator isim = theMuonHits.begin();
	isim != theMuonHits.end(); ++isim){
     std::cout << (*isim) << " Track Id = " << isim->trackId() << std::endl;
   }

   std::cout << "\n Calorimeter Hits in the event = " << theCaloHits.size() << std::endl; 
   std::cout << "\n" << std::endl;
   for (std::vector<PCaloHit>::iterator isim = theCaloHits.begin();
	isim != theCaloHits.end(); ++isim){
     std::cout << (*isim) << std::endl;
   }

   return;

}


//define this as a plug-in
DEFINE_FWK_MODULE(SimHitCaloHitDumper);
