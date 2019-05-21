// system include files
#include <vector>
#include <string>
#include <iostream>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

class SimHitCaloHitDumper : public edm::EDAnalyzer {
public:
  SimHitCaloHitDumper(const edm::ParameterSet& iConfig);
  ~SimHitCaloHitDumper() override{};

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override{};
  void endJob() override{};

private:
  std::string moduleName;

  edm::EDGetTokenT<edm::PSimHitContainer> PixelBarrelLowTofToken;
  edm::EDGetTokenT<edm::PSimHitContainer> PixelBarrelHighTofToken;
  edm::EDGetTokenT<edm::PSimHitContainer> PixelEndcapLowTofToken;
  edm::EDGetTokenT<edm::PSimHitContainer> PixelEndcapHighTofToken;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTIBLowTofToken;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTIBHighTofToken;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTIDLowTofToken;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTIDHighTofToken;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTOBLowTofToken;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTOBHighTofToken;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTECLowTofToken;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTECHighTofToken;

  edm::EDGetTokenT<edm::PSimHitContainer> MuonDTToken;
  edm::EDGetTokenT<edm::PSimHitContainer> MuonCSCToken;
  edm::EDGetTokenT<edm::PSimHitContainer> MuonRPCToken;

  edm::EDGetTokenT<edm::PSimHitContainer> FastTimerBTLToken;
  edm::EDGetTokenT<edm::PSimHitContainer> FastTimerETLToken;

  edm::EDGetTokenT<edm::PCaloHitContainer> EcalEBToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> EcalEEToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> EcalESToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> HcalToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> CaloTkToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> ZDCToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> CastorTUToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> CastorPLToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> CastorFIToken;
  edm::EDGetTokenT<edm::PCaloHitContainer> CastorBUToken;
};

SimHitCaloHitDumper::SimHitCaloHitDumper(const edm::ParameterSet& iConfig)
    : moduleName(iConfig.getParameter<std::string>("moduleName")) {
  PixelBarrelLowTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsPixelBarrelLowTof"));
  PixelBarrelHighTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsPixelBarrelHighTof"));
  PixelEndcapLowTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsPixelEndcapLowTof"));
  PixelEndcapHighTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsPixelEndcapHighTof"));
  TrackerTIBLowTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTIBLowTof"));
  TrackerTIBHighTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTIBHighTof"));
  TrackerTIDLowTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTIDLowTof"));
  TrackerTIDHighTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTIDHighTof"));
  TrackerTOBLowTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTOBLowTof"));
  TrackerTOBHighTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTOBHighTof"));
  TrackerTECLowTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTECLowTof"));
  TrackerTECHighTofToken =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTECHighTof"));

  MuonDTToken = consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "MuonDTHits"));
  MuonCSCToken = consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "MuonCSCHits"));
  MuonRPCToken = consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "MuonRPCHits"));

  FastTimerBTLToken = consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "FastTimerHitsBarrel"));
  FastTimerETLToken = consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "FastTimerHitsEndcap"));

  EcalEBToken = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "EBHits"));
  EcalEEToken = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "EEHits"));
  EcalESToken = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "ESHits"));
  HcalToken = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "HcalHits"));
  CaloTkToken = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "CaloTkHits"));
  ZDCToken = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "ZDC"));
  CastorTUToken = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "CastorTUHits"));
  CastorPLToken = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "CastorPLHits"));
  CastorFIToken = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "CastorFIHits"));
  CastorBUToken = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "CastorBUHits"));
}

void SimHitCaloHitDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::vector<PSimHit> theTrackerHits;
  std::vector<PSimHit> theMuonHits;
  std::vector<PSimHit> theMTDHits;
  std::vector<PCaloHit> theCaloHits;

  std::vector<std::pair<int, std::string> > theTrackerComposition;
  std::vector<std::pair<int, std::string> > theMuonComposition;
  std::vector<std::pair<int, std::string> > theMTDComposition;
  std::vector<std::pair<int, std::string> > theCaloComposition;

  edm::Handle<edm::PSimHitContainer> PixelBarrelHitsLowTof;
  edm::Handle<edm::PSimHitContainer> PixelBarrelHitsHighTof;
  edm::Handle<edm::PSimHitContainer> PixelEndcapHitsLowTof;
  edm::Handle<edm::PSimHitContainer> PixelEndcapHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TIBHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TIBHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TIDHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TIDHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TOBHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TOBHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TECHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TECHitsHighTof;

  edm::Handle<edm::PSimHitContainer> DTHits;
  edm::Handle<edm::PSimHitContainer> CSCHits;
  edm::Handle<edm::PSimHitContainer> RPCHits;

  edm::Handle<edm::PSimHitContainer> BTLHits;
  edm::Handle<edm::PSimHitContainer> ETLHits;

  edm::Handle<edm::PCaloHitContainer> EBHits;
  edm::Handle<edm::PCaloHitContainer> EEHits;
  edm::Handle<edm::PCaloHitContainer> ESHits;
  edm::Handle<edm::PCaloHitContainer> HcalHits;
  edm::Handle<edm::PCaloHitContainer> CaloTkHits;
  edm::Handle<edm::PCaloHitContainer> ZDCHits;
  edm::Handle<edm::PCaloHitContainer> CastorTUHits;
  edm::Handle<edm::PCaloHitContainer> CastorPLHits;
  edm::Handle<edm::PCaloHitContainer> CastorFIHits;
  edm::Handle<edm::PCaloHitContainer> CastorBUHits;

  iEvent.getByToken(PixelBarrelLowTofToken, PixelBarrelHitsLowTof);
  iEvent.getByToken(PixelBarrelHighTofToken, PixelBarrelHitsHighTof);
  iEvent.getByToken(PixelEndcapLowTofToken, PixelEndcapHitsLowTof);
  iEvent.getByToken(PixelEndcapHighTofToken, PixelEndcapHitsHighTof);
  iEvent.getByToken(TrackerTIBLowTofToken, TIBHitsLowTof);
  iEvent.getByToken(TrackerTIBHighTofToken, TIBHitsHighTof);
  iEvent.getByToken(TrackerTIDLowTofToken, TIDHitsLowTof);
  iEvent.getByToken(TrackerTIDHighTofToken, TIDHitsHighTof);
  iEvent.getByToken(TrackerTOBLowTofToken, TOBHitsLowTof);
  iEvent.getByToken(TrackerTOBHighTofToken, TOBHitsHighTof);
  iEvent.getByToken(TrackerTECLowTofToken, TECHitsLowTof);
  iEvent.getByToken(TrackerTECHighTofToken, TECHitsHighTof);

  iEvent.getByToken(MuonDTToken, DTHits);
  iEvent.getByToken(MuonCSCToken, CSCHits);
  iEvent.getByToken(MuonRPCToken, RPCHits);

  iEvent.getByToken(EcalEBToken, EBHits);
  iEvent.getByToken(EcalEEToken, EEHits);
  iEvent.getByToken(EcalESToken, ESHits);
  iEvent.getByToken(HcalToken, HcalHits);
  iEvent.getByToken(CaloTkToken, CaloTkHits);
  iEvent.getByToken(ZDCToken, ZDCHits);
  iEvent.getByToken(CastorTUToken, CastorTUHits);
  iEvent.getByToken(CastorPLToken, CastorPLHits);
  iEvent.getByToken(CastorFIToken, CastorFIHits);
  iEvent.getByToken(CastorBUToken, CastorBUHits);

  iEvent.getByToken(FastTimerBTLToken, BTLHits);
  iEvent.getByToken(FastTimerETLToken, ETLHits);

  int oldsize = 0;

  if (PixelBarrelHitsLowTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), PixelBarrelHitsLowTof->begin(), PixelBarrelHitsLowTof->end());
    std::pair<int, std::string> label1(theTrackerHits.size(), "PixelBarrelHitsLowTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label1);
  }
  if (PixelBarrelHitsHighTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), PixelBarrelHitsHighTof->begin(), PixelBarrelHitsHighTof->end());
    std::pair<int, std::string> label2(theTrackerHits.size() - oldsize, "PixelBarrelHitsHighTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label2);
  }
  if (PixelEndcapHitsLowTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), PixelEndcapHitsLowTof->begin(), PixelEndcapHitsLowTof->end());
    std::pair<int, std::string> label3(theTrackerHits.size() - oldsize, "PixelEndcapHitsLowTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label3);
  }
  if (PixelEndcapHitsHighTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), PixelEndcapHitsHighTof->begin(), PixelEndcapHitsHighTof->end());
    std::pair<int, std::string> label4(theTrackerHits.size() - oldsize, "PixelEndcapHitsHighTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label4);
  }
  if (TIBHitsLowTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), TIBHitsLowTof->begin(), TIBHitsLowTof->end());
    std::pair<int, std::string> label5(theTrackerHits.size() - oldsize, "TIBHitsLowTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label5);
  }
  if (TIBHitsHighTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), TIBHitsHighTof->begin(), TIBHitsHighTof->end());
    std::pair<int, std::string> label6(theTrackerHits.size() - oldsize, "TIBHitsHighTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label6);
  }
  if (TIDHitsLowTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), TIDHitsLowTof->begin(), TIDHitsLowTof->end());
    std::pair<int, std::string> label7(theTrackerHits.size() - oldsize, "TIDHitsLowTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label7);
  }
  if (TIDHitsHighTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), TIDHitsHighTof->begin(), TIDHitsHighTof->end());
    std::pair<int, std::string> label8(theTrackerHits.size() - oldsize, "TIDHitsHighTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label8);
  }
  if (TOBHitsLowTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), TOBHitsLowTof->begin(), TOBHitsLowTof->end());
    std::pair<int, std::string> label9(theTrackerHits.size() - oldsize, "TOBHitsLowTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label9);
  }
  if (TOBHitsHighTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), TOBHitsHighTof->begin(), TOBHitsHighTof->end());
    std::pair<int, std::string> label10(theTrackerHits.size() - oldsize, "TOBHitsHighTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label10);
  }
  if (TECHitsLowTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), TECHitsLowTof->begin(), TECHitsLowTof->end());
    std::pair<int, std::string> label11(theTrackerHits.size() - oldsize, "TECHitsLowTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label11);
  }
  if (TECHitsHighTof.isValid()) {
    theTrackerHits.insert(theTrackerHits.end(), TECHitsHighTof->begin(), TECHitsHighTof->end());
    std::pair<int, std::string> label12(theTrackerHits.size() - oldsize, "TECHitsHighTof");
    oldsize = theTrackerHits.size();
    theTrackerComposition.push_back(label12);
  }

  oldsize = 0;
  if (DTHits.isValid()) {
    theMuonHits.insert(theMuonHits.end(), DTHits->begin(), DTHits->end());
    std::pair<int, std::string> label13(theMuonHits.size() - oldsize, "DTHits");
    oldsize = theMuonHits.size();
    theMuonComposition.push_back(label13);
  }
  if (CSCHits.isValid()) {
    theMuonHits.insert(theMuonHits.end(), CSCHits->begin(), CSCHits->end());
    std::pair<int, std::string> label14(theMuonHits.size() - oldsize, "CSCHits");
    oldsize = theMuonHits.size();
    theMuonComposition.push_back(label14);
  }
  if (RPCHits.isValid()) {
    theMuonHits.insert(theMuonHits.end(), RPCHits->begin(), RPCHits->end());
    std::pair<int, std::string> label15(theMuonHits.size() - oldsize, "RPCHits");
    oldsize = theMuonHits.size();
    theMuonComposition.push_back(label15);
  }

  oldsize = 0;
  if (EBHits.isValid()) {
    theCaloHits.insert(theCaloHits.end(), EBHits->begin(), EBHits->end());
    std::pair<int, std::string> label16(theCaloHits.size() - oldsize, "EBHits");
    oldsize = theCaloHits.size();
    theCaloComposition.push_back(label16);
  }
  if (EEHits.isValid()) {
    theCaloHits.insert(theCaloHits.end(), EEHits->begin(), EEHits->end());
    std::pair<int, std::string> label17(theCaloHits.size() - oldsize, "EEHits");
    oldsize = theCaloHits.size();
    theCaloComposition.push_back(label17);
  }
  if (ESHits.isValid()) {
    theCaloHits.insert(theCaloHits.end(), ESHits->begin(), ESHits->end());
    std::pair<int, std::string> label18(theCaloHits.size() - oldsize, "ESHits");
    oldsize = theCaloHits.size();
    theCaloComposition.push_back(label18);
  }
  if (HcalHits.isValid()) {
    theCaloHits.insert(theCaloHits.end(), HcalHits->begin(), HcalHits->end());
    std::pair<int, std::string> label19(theCaloHits.size() - oldsize, "HcalHits");
    oldsize = theCaloHits.size();
    theCaloComposition.push_back(label19);
  }
  if (CaloTkHits.isValid()) {
    theCaloHits.insert(theCaloHits.end(), CaloTkHits->begin(), CaloTkHits->end());
    std::pair<int, std::string> label20(theCaloHits.size() - oldsize, "CaloTkHits");
    oldsize = theCaloHits.size();
    theCaloComposition.push_back(label20);
  }
  if (ZDCHits.isValid()) {
    theCaloHits.insert(theCaloHits.end(), ZDCHits->begin(), ZDCHits->end());
    std::pair<int, std::string> label21(theCaloHits.size() - oldsize, "ZDCHITS");
    oldsize = theCaloHits.size();
    theCaloComposition.push_back(label21);
  }
  if (CastorTUHits.isValid()) {
    theCaloHits.insert(theCaloHits.end(), CastorTUHits->begin(), CastorTUHits->end());
    std::pair<int, std::string> label22(theCaloHits.size() - oldsize, "CastorTU");
    oldsize = theCaloHits.size();
    theCaloComposition.push_back(label22);
  }
  if (CastorPLHits.isValid()) {
    theCaloHits.insert(theCaloHits.end(), CastorPLHits->begin(), CastorPLHits->end());
    std::pair<int, std::string> label23(theCaloHits.size() - oldsize, "CastorPL");
    oldsize = theCaloHits.size();
    theCaloComposition.push_back(label23);
  }
  if (CastorFIHits.isValid()) {
    theCaloHits.insert(theCaloHits.end(), CastorFIHits->begin(), CastorFIHits->end());
    std::pair<int, std::string> label24(theCaloHits.size() - oldsize, "CastorFI");
    oldsize = theCaloHits.size();
    theCaloComposition.push_back(label24);
  }
  if (CastorBUHits.isValid()) {
    theCaloHits.insert(theCaloHits.end(), CastorBUHits->begin(), CastorBUHits->end());
    std::pair<int, std::string> label25(theCaloHits.size() - oldsize, "CastorBU");
    oldsize = theCaloHits.size();
    theCaloComposition.push_back(label25);
  }

  oldsize = 0;
  if (BTLHits.isValid()) {
    theMTDHits.insert(theMTDHits.end(), BTLHits->begin(), BTLHits->end());
    std::pair<int, std::string> label26(theMTDHits.size() - oldsize, "BTLHits");
    oldsize = theMTDHits.size();
    theMTDComposition.push_back(label26);
  }
  if (ETLHits.isValid()) {
    theMTDHits.insert(theMTDHits.end(), ETLHits->begin(), ETLHits->end());
    std::pair<int, std::string> label27(theMTDHits.size() - oldsize, "ETLHits");
    oldsize = theMTDHits.size();
    theMTDComposition.push_back(label27);
  }

  std::cout << "\n SimHit / CaloHit structure dump \n" << std::endl;
  std::cout << " Tracker Hits in the event = " << theTrackerHits.size() << std::endl;
  std::cout << "\n" << std::endl;
  //   for (std::vector<PSimHit>::iterator isim = theTrackerHits.begin();
  //      isim != theTrackerHits.end(); ++isim){
  //     std::cout << (*isim) << " Track Id = " << isim->trackId() << std::endl;
  //   }
  int nhit = 0;
  for (std::vector<std::pair<int, std::string> >::iterator icoll = theTrackerComposition.begin();
       icoll != theTrackerComposition.end();
       ++icoll) {
    std::cout << "\n" << std::endl;
    std::cout << (*icoll).second << " hits in the event = " << (*icoll).first << std::endl;
    std::cout << "\n" << std::endl;
    for (int ihit = 0; ihit < (*icoll).first; ++ihit) {
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
  for (std::vector<std::pair<int, std::string> >::iterator icoll = theMuonComposition.begin();
       icoll != theMuonComposition.end();
       ++icoll) {
    std::cout << "\n" << std::endl;
    std::cout << (*icoll).second << " hits in the event = " << (*icoll).first << std::endl;
    std::cout << "\n" << std::endl;
    for (int ihit = 0; ihit < (*icoll).first; ++ihit) {
      std::cout << theMuonHits[nhit] << " Track Id = " << theMuonHits[nhit].trackId() << std::endl;
      nhit++;
    }
  }

  std::cout << "\n MTD Hits in the event = " << theMTDHits.size() << std::endl;
  std::cout << "\n" << std::endl;
  //   for (std::vector<PSimHit>::iterator isim = theMTDHits.begin();
  //        isim != theMTDHits.end(); ++isim){
  //     std::cout << (*isim) << " Track Id = " << isim->trackId() << std::endl;
  //   }
  nhit = 0;
  for (std::vector<std::pair<int, std::string> >::iterator icoll = theMTDComposition.begin();
       icoll != theMTDComposition.end();
       ++icoll) {
    std::cout << "\n" << std::endl;
    std::cout << (*icoll).second << " hits in the event = " << (*icoll).first << std::endl;
    std::cout << "\n" << std::endl;
    for (int ihit = 0; ihit < (*icoll).first; ++ihit) {
      std::cout << theMTDHits[nhit] << " Track Id = " << theMTDHits[nhit].trackId() << std::endl;
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
  for (std::vector<std::pair<int, std::string> >::iterator icoll = theCaloComposition.begin();
       icoll != theCaloComposition.end();
       ++icoll) {
    std::cout << "\n" << std::endl;
    std::cout << (*icoll).second << " hits in the event = " << (*icoll).first << std::endl;
    std::cout << "\n" << std::endl;
    for (int ihit = 0; ihit < (*icoll).first; ++ihit) {
      std::cout << theCaloHits[nhit] << std::endl;
      nhit++;
    }
  }

  return;
}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(SimHitCaloHitDumper);
