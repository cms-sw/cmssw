// system include files
#include <memory>
#include <vector>
#include <string>

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

   std::vector< std::pair<int,std::string> > theTrackerComposition;
   std::vector< std::pair<int,std::string> > theMuonComposition;
   std::vector< std::pair<int,std::string> > theCaloComposition;
   

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
   Handle<PCaloHitContainer> ZDCHits;
   Handle<PCaloHitContainer> CastorTUHits;
   Handle<PCaloHitContainer> CastorPLHits;
   Handle<PCaloHitContainer> CastorFIHits;
   Handle<PCaloHitContainer> CastorBUHits;

   iEvent.getByLabel(edm::InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof",processName), PixelBarrelHitsLowTof);
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
   iEvent.getByLabel("g4SimHits","ZDCHITS", ZDCHits );
   iEvent.getByLabel("g4SimHits","CastorTU", CastorTUHits );
   iEvent.getByLabel("g4SimHits","CastorPL", CastorPLHits );
   iEvent.getByLabel("g4SimHits","CastorFI", CastorFIHits );
   iEvent.getByLabel("g4SimHits","CastorBU", CastorBUHits );

   int oldsize = 0;

   if ( PixelBarrelHitsLowTof.isValid() ) {
     theTrackerHits.insert(theTrackerHits.end(), PixelBarrelHitsLowTof->begin(), PixelBarrelHitsLowTof->end()); 
     std::pair<int,std::string> label1(theTrackerHits.size(),"PixelBarrelHitsLowTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label1);
   }
   if ( PixelBarrelHitsHighTof.isValid() ) {
     theTrackerHits.insert(theTrackerHits.end(), PixelBarrelHitsHighTof->begin(), PixelBarrelHitsHighTof->end());
     std::pair<int,std::string> label2(theTrackerHits.size()-oldsize,"PixelBarrelHitsHighTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label2);
   }
   if ( PixelEndcapHitsLowTof.isValid() ) { 
     theTrackerHits.insert(theTrackerHits.end(), PixelEndcapHitsLowTof->begin(), PixelEndcapHitsLowTof->end()); 
     std::pair<int,std::string> label3(theTrackerHits.size()-oldsize,"PixelEndcapHitsLowTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label3);
   }
   if ( PixelEndcapHitsHighTof.isValid() ) {
     theTrackerHits.insert(theTrackerHits.end(), PixelEndcapHitsHighTof->begin(), PixelEndcapHitsHighTof->end());
     std::pair<int,std::string> label4(theTrackerHits.size()-oldsize,"PixelEndcapHitsHighTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label4);
   }
   if ( TIBHitsLowTof.isValid() ) {
     theTrackerHits.insert(theTrackerHits.end(), TIBHitsLowTof->begin(), TIBHitsLowTof->end()); 
     std::pair<int,std::string> label5(theTrackerHits.size()-oldsize,"TIBHitsLowTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label5);
   }
   if ( TIBHitsHighTof.isValid() ) {
     theTrackerHits.insert(theTrackerHits.end(), TIBHitsHighTof->begin(), TIBHitsHighTof->end());
     std::pair<int,std::string> label6(theTrackerHits.size()-oldsize,"TIBHitsHighTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label6);
   }
   if ( TIDHitsLowTof.isValid() ) {
     theTrackerHits.insert(theTrackerHits.end(), TIDHitsLowTof->begin(), TIDHitsLowTof->end()); 
     std::pair<int,std::string> label7(theTrackerHits.size()-oldsize,"TIDHitsLowTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label7);
   }
   if ( TIDHitsHighTof.isValid() ) {
     theTrackerHits.insert(theTrackerHits.end(), TIDHitsHighTof->begin(), TIDHitsHighTof->end());
     std::pair<int,std::string> label8(theTrackerHits.size()-oldsize,"TIDHitsHighTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label8);
   }
   if ( TOBHitsLowTof.isValid() ) {
     theTrackerHits.insert(theTrackerHits.end(), TOBHitsLowTof->begin(), TOBHitsLowTof->end()); 
     std::pair<int,std::string> label9(theTrackerHits.size()-oldsize,"TOBHitsLowTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label9);
   }
   if ( TOBHitsHighTof.isValid() ) {
     theTrackerHits.insert(theTrackerHits.end(), TOBHitsHighTof->begin(), TOBHitsHighTof->end());
     std::pair<int,std::string> label10(theTrackerHits.size()-oldsize,"TOBHitsHighTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label10);
   }
   if ( TECHitsLowTof.isValid() ) {
     theTrackerHits.insert(theTrackerHits.end(), TECHitsLowTof->begin(), TECHitsLowTof->end()); 
     std::pair<int,std::string> label11(theTrackerHits.size()-oldsize,"TECHitsLowTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label11);
   }
   if ( TECHitsHighTof.isValid() ) {
     theTrackerHits.insert(theTrackerHits.end(), TECHitsHighTof->begin(), TECHitsHighTof->end());
     std::pair<int,std::string> label12(theTrackerHits.size()-oldsize,"TECHitsHighTof");
     oldsize = theTrackerHits.size();
     theTrackerComposition.push_back(label12);
   }

   oldsize = 0;
   if ( DTHits.isValid() ) {
     theMuonHits.insert(theMuonHits.end(), DTHits->begin(), DTHits->end() );
     std::pair<int,std::string> label13(theMuonHits.size()-oldsize,"DTHits");
     oldsize = theMuonHits.size();
     theMuonComposition.push_back(label13);
   }
   if ( CSCHits.isValid() ) {
     theMuonHits.insert(theMuonHits.end(), CSCHits->begin(), CSCHits->end() );
     std::pair<int,std::string> label14(theMuonHits.size()-oldsize,"CSCHits");
     oldsize = theMuonHits.size();
     theMuonComposition.push_back(label14);
   }
   if ( RPCHits.isValid() ) {
     theMuonHits.insert(theMuonHits.end(), RPCHits->begin(), RPCHits->end() );
     std::pair<int,std::string> label15(theMuonHits.size()-oldsize,"RPCHits");
     oldsize = theMuonHits.size();
     theMuonComposition.push_back(label15);
   }

   oldsize = 0;
   if ( EBHits.isValid() ) {
     theCaloHits.insert(theCaloHits.end(), EBHits->begin(), EBHits->end() );
     std::pair<int,std::string> label16(theCaloHits.size()-oldsize,"EBHits");
     oldsize = theCaloHits.size();
     theCaloComposition.push_back(label16);
   }
   if ( EEHits.isValid() ) {
     theCaloHits.insert(theCaloHits.end(), EEHits->begin(), EEHits->end() );
     std::pair<int,std::string> label17(theCaloHits.size()-oldsize,"EEHits");
     oldsize = theCaloHits.size();
     theCaloComposition.push_back(label17);
   }
   if ( ESHits.isValid() ) {
     theCaloHits.insert(theCaloHits.end(), ESHits->begin(), ESHits->end() );
     std::pair<int,std::string> label18(theCaloHits.size()-oldsize,"ESHits");
     oldsize = theCaloHits.size();
     theCaloComposition.push_back(label18);
   }
   if ( HcalHits.isValid() ) {
     theCaloHits.insert(theCaloHits.end(), HcalHits->begin(), HcalHits->end() );
     std::pair<int,std::string> label19(theCaloHits.size()-oldsize,"HcalHits");
     oldsize = theCaloHits.size();
     theCaloComposition.push_back(label19);
   }
   if ( CaloTkHits.isValid() ) {
     theCaloHits.insert(theCaloHits.end(), CaloTkHits->begin(), CaloTkHits->end() );
     std::pair<int,std::string> label20(theCaloHits.size()-oldsize,"CaloTkHits");
     oldsize = theCaloHits.size();
     theCaloComposition.push_back(label20);
   } 
   if ( ZDCHits.isValid() ) {
     theCaloHits.insert(theCaloHits.end(), ZDCHits->begin(), ZDCHits->end() );
     std::pair<int,std::string> label21(theCaloHits.size()-oldsize,"ZDCHITS");
     oldsize = theCaloHits.size();
     theCaloComposition.push_back(label21);
   } 
   if ( CastorTUHits.isValid() ) {
     theCaloHits.insert(theCaloHits.end(), CastorTUHits->begin(), CastorTUHits->end() );
     std::pair<int,std::string> label22(theCaloHits.size()-oldsize,"CastorTU");
     oldsize = theCaloHits.size();
     theCaloComposition.push_back(label22);
   } 
   if ( CastorPLHits.isValid() ) {
     theCaloHits.insert(theCaloHits.end(), CastorPLHits->begin(), CastorPLHits->end() );
     std::pair<int,std::string> label23(theCaloHits.size()-oldsize,"CastorPL");
     oldsize = theCaloHits.size();
     theCaloComposition.push_back(label23);
   } 
   if ( CastorFIHits.isValid() ) {
     theCaloHits.insert(theCaloHits.end(), CastorFIHits->begin(), CastorFIHits->end() );
     std::pair<int,std::string> label24(theCaloHits.size()-oldsize,"CastorFI");
     oldsize = theCaloHits.size();
     theCaloComposition.push_back(label24);
   } 
   if ( CastorBUHits.isValid() ) {
     theCaloHits.insert(theCaloHits.end(), CastorBUHits->begin(), CastorBUHits->end() );
     std::pair<int,std::string> label25(theCaloHits.size()-oldsize,"CastorBU");
     oldsize = theCaloHits.size();
     theCaloComposition.push_back(label25);
   } 

   std::cout << "\n SimHit / CaloHit structure dump \n" << std::endl;
   std::cout << " Tracker Hits in the event = " << theTrackerHits.size() << std::endl; 
   std::cout << "\n" << std::endl;
   //   for (std::vector<PSimHit>::iterator isim = theTrackerHits.begin();
   //      isim != theTrackerHits.end(); ++isim){
   //     std::cout << (*isim) << " Track Id = " << isim->trackId() << std::endl;
   //   }
   int nhit = 0;
   for (std::vector< std::pair<int,std::string> >::iterator icoll = theTrackerComposition.begin();
        icoll != theTrackerComposition.end(); ++icoll){
     std::cout << "\n" << std::endl;
     std::cout << (*icoll).second << " hits in the event = " << (*icoll).first << std::endl;
     std::cout << "\n" << std::endl;
     for ( int ihit = 0; ihit < (*icoll).first; ++ihit ) {
       std::cout << theTrackerHits[nhit] << " Track Id = " << theTrackerHits[nhit].trackId() << std::endl;
       nhit++;
     }
   }   

   std::cout << "\n Muon Hits in the event = " << theMuonHits.size() << std::endl; 
   std::cout << "\n" << std::endl;
   //   for (std::vector<PSimHit>::iterator isim = theMuonHits.begin();
   //        isim != theMuonHits.end(); ++isim){
   //     std::cout << (*isim) << " Track Id = " << isim->trackId() << std::endl;
   //   }
   nhit = 0;
   for (std::vector< std::pair<int,std::string> >::iterator icoll = theMuonComposition.begin();
        icoll != theMuonComposition.end(); ++icoll){
     std::cout << "\n" << std::endl;
     std::cout << (*icoll).second << " hits in the event = " << (*icoll).first << std::endl;
     std::cout << "\n" << std::endl;
     for ( int ihit = 0; ihit < (*icoll).first; ++ihit ) {
       std::cout << theMuonHits[nhit] << " Track Id = " << theMuonHits[nhit].trackId() << std::endl;
       nhit++;
     }
   }   

   std::cout << "\n Calorimeter Hits in the event = " << theCaloHits.size() << std::endl; 
   std::cout << "\n" << std::endl;
   //   for (std::vector<PCaloHit>::iterator isim = theCaloHits.begin();
   //        isim != theCaloHits.end(); ++isim){
   //     std::cout << (*isim) << std::endl;
   //   }
   nhit = 0;
   for (std::vector< std::pair<int,std::string> >::iterator icoll = theCaloComposition.begin();
        icoll != theCaloComposition.end(); ++icoll){
     std::cout << "\n" << std::endl;
     std::cout << (*icoll).second << " hits in the event = " << (*icoll).first << std::endl;
     std::cout << "\n" << std::endl;
     for ( int ihit = 0; ihit < (*icoll).first; ++ihit ) {
       std::cout << theCaloHits[nhit] << std::endl;
       nhit++;
     }
   }   

   return;

}


//define this as a plug-in
DEFINE_FWK_MODULE(SimHitCaloHitDumper);
