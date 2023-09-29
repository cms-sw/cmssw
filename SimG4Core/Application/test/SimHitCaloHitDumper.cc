// system include files
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

class SimHitCaloHitDumper : public edm::one::EDAnalyzer<> {
public:
  SimHitCaloHitDumper(const edm::ParameterSet&);
  ~SimHitCaloHitDumper() override{};

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override{};
  void endJob() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::string moduleName;

  edm::EDGetTokenT<edm::PSimHitContainer> PixelBarrelLowTofToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> PixelBarrelHighTofToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> PixelEndcapLowTofToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> PixelEndcapHighTofToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTIBLowTofToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTIBHighTofToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTIDLowTofToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTIDHighTofToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTOBLowTofToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTOBHighTofToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTECLowTofToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> TrackerTECHighTofToken_;

  edm::EDGetTokenT<edm::PSimHitContainer> MuonDTToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> MuonCSCToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> MuonRPCToken_;

  edm::EDGetTokenT<edm::PSimHitContainer> FastTimerBTLToken_;
  edm::EDGetTokenT<edm::PSimHitContainer> FastTimerETLToken_;

  edm::EDGetTokenT<edm::PCaloHitContainer> EcalEBToken_;
  edm::EDGetTokenT<edm::PCaloHitContainer> EcalEEToken_;
  edm::EDGetTokenT<edm::PCaloHitContainer> EcalESToken_;
  edm::EDGetTokenT<edm::PCaloHitContainer> HcalToken_;
  edm::EDGetTokenT<edm::PCaloHitContainer> CaloTkToken_;
  edm::EDGetTokenT<edm::PCaloHitContainer> ZDCToken_;
  edm::EDGetTokenT<edm::PCaloHitContainer> CastorTUToken_;
  edm::EDGetTokenT<edm::PCaloHitContainer> CastorPLToken_;
  edm::EDGetTokenT<edm::PCaloHitContainer> CastorFIToken_;
  edm::EDGetTokenT<edm::PCaloHitContainer> CastorBUToken_;
};

SimHitCaloHitDumper::SimHitCaloHitDumper(const edm::ParameterSet& iConfig) {
  PixelBarrelLowTofToken_ =
      consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsPixelBarrelLowTof"));
  PixelBarrelHighTofToken_ =
      consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsPixelBarrelHighTof"));
  PixelEndcapLowTofToken_ =
      consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsPixelEndcapLowTof"));
  PixelEndcapHighTofToken_ =
      consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsPixelEndcapHighTof"));
  TrackerTIBLowTofToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsTIBLowTof"));
  TrackerTIBHighTofToken_ =
      consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsTIBHighTof"));
  TrackerTIDLowTofToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsTIDLowTof"));
  TrackerTIDHighTofToken_ =
      consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsTIDHighTof"));
  TrackerTOBLowTofToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsTOBLowTof"));
  TrackerTOBHighTofToken_ =
      consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsTOBHighTof"));
  TrackerTECLowTofToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsTECLowTof"));
  TrackerTECHighTofToken_ =
      consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("TrackerHitsTECHighTof"));

  MuonDTToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("MuonDTHits"));
  MuonCSCToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("MuonCSCHits"));
  MuonRPCToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("MuonRPCHits"));

  FastTimerBTLToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("FastTimerHitsBarrel"));
  FastTimerETLToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("FastTimerHitsEndcap"));

  EcalEBToken_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("EcalHitsEB"));
  EcalEEToken_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("EcalHitsEE"));
  EcalESToken_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("EcalHitsES"));
  HcalToken_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("HcalHits"));
  CaloTkToken_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("CaloHitsTk"));
  ZDCToken_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("ZDCHITS"));
  CastorTUToken_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("CastorTU"));
  CastorPLToken_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("CastorPL"));
  CastorFIToken_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("CastorFI"));
  CastorBUToken_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("CastorBU"));
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

  auto PixelBarrelHitsLowTof = iEvent.getHandle(PixelBarrelLowTofToken_);
  auto PixelBarrelHitsHighTof = iEvent.getHandle(PixelBarrelHighTofToken_);
  auto PixelEndcapHitsLowTof = iEvent.getHandle(PixelEndcapLowTofToken_);
  auto PixelEndcapHitsHighTof = iEvent.getHandle(PixelEndcapHighTofToken_);
  auto TIBHitsLowTof = iEvent.getHandle(TrackerTIBLowTofToken_);
  auto TIBHitsHighTof = iEvent.getHandle(TrackerTIBHighTofToken_);
  auto TIDHitsLowTof = iEvent.getHandle(TrackerTIDLowTofToken_);
  auto TIDHitsHighTof = iEvent.getHandle(TrackerTIDHighTofToken_);
  auto TOBHitsLowTof = iEvent.getHandle(TrackerTOBLowTofToken_);
  auto TOBHitsHighTof = iEvent.getHandle(TrackerTOBHighTofToken_);
  auto TECHitsLowTof = iEvent.getHandle(TrackerTECLowTofToken_);
  auto TECHitsHighTof = iEvent.getHandle(TrackerTECHighTofToken_);

  auto DTHits = iEvent.getHandle(MuonDTToken_);
  auto CSCHits = iEvent.getHandle(MuonCSCToken_);
  auto RPCHits = iEvent.getHandle(MuonRPCToken_);

  auto EBHits = iEvent.getHandle(EcalEBToken_);
  auto EEHits = iEvent.getHandle(EcalEEToken_);
  auto ESHits = iEvent.getHandle(EcalESToken_);
  auto HcalHits = iEvent.getHandle(HcalToken_);
  auto CaloTkHits = iEvent.getHandle(CaloTkToken_);
  auto ZDCHits = iEvent.getHandle(ZDCToken_);
  auto CastorTUHits = iEvent.getHandle(CastorTUToken_);
  auto CastorPLHits = iEvent.getHandle(CastorPLToken_);
  auto CastorFIHits = iEvent.getHandle(CastorFIToken_);
  auto CastorBUHits = iEvent.getHandle(CastorBUToken_);

  auto BTLHits = iEvent.getHandle(FastTimerBTLToken_);
  auto ETLHits = iEvent.getHandle(FastTimerETLToken_);

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

  edm::LogPrint("SimHitCaloHitDumper") << "\n SimHit / CaloHit structure dump \n";
  edm::LogPrint("SimHitCaloHitDumper") << " Tracker Hits in the event = " << theTrackerHits.size();
  edm::LogPrint("SimHitCaloHitDumper") << "\n";
  //   for (std::vector<PSimHit>::iterator isim = theTrackerHits.begin();
  //      isim != theTrackerHits.end(); ++isim){
  //     edm::LogPrint("SimHitCaloHitDumper") << (*isim) << " Track Id = " << isim->trackId() ;
  //   }
  int nhit = 0;
  for (std::vector<std::pair<int, std::string> >::iterator icoll = theTrackerComposition.begin();
       icoll != theTrackerComposition.end();
       ++icoll) {
    edm::LogPrint("SimHitCaloHitDumper") << "\n";
    edm::LogPrint("SimHitCaloHitDumper") << (*icoll).second << " hits in the event = " << (*icoll).first;
    edm::LogPrint("SimHitCaloHitDumper") << "\n";
    for (int ihit = 0; ihit < (*icoll).first; ++ihit) {
      edm::LogPrint("SimHitCaloHitDumper") << theTrackerHits[nhit] << " Track Id = " << theTrackerHits[nhit].trackId();
      nhit++;
    }
  }

  edm::LogPrint("SimHitCaloHitDumper") << "\n Muon Hits in the event = " << theMuonHits.size();
  edm::LogPrint("SimHitCaloHitDumper") << "\n";
  //   for (std::vector<PSimHit>::iterator isim = theMuonHits.begin();
  //        isim != theMuonHits.end(); ++isim){
  //     edm::LogPrint("SimHitCaloHitDumper") << (*isim) << " Track Id = " << isim->trackId() ;
  //   }
  nhit = 0;
  for (std::vector<std::pair<int, std::string> >::iterator icoll = theMuonComposition.begin();
       icoll != theMuonComposition.end();
       ++icoll) {
    edm::LogPrint("SimHitCaloHitDumper") << "\n";
    edm::LogPrint("SimHitCaloHitDumper") << (*icoll).second << " hits in the event = " << (*icoll).first;
    edm::LogPrint("SimHitCaloHitDumper") << "\n";
    for (int ihit = 0; ihit < (*icoll).first; ++ihit) {
      edm::LogPrint("SimHitCaloHitDumper") << theMuonHits[nhit] << " Track Id = " << theMuonHits[nhit].trackId();
      nhit++;
    }
  }

  edm::LogPrint("SimHitCaloHitDumper") << "\n MTD Hits in the event = " << theMTDHits.size();
  edm::LogPrint("SimHitCaloHitDumper") << "\n";
  //   for (std::vector<PSimHit>::iterator isim = theMTDHits.begin();
  //        isim != theMTDHits.end(); ++isim){
  //     edm::LogPrint("SimHitCaloHitDumper") << (*isim) << " Track Id = " << isim->trackId() ;
  //   }
  nhit = 0;
  for (std::vector<std::pair<int, std::string> >::iterator icoll = theMTDComposition.begin();
       icoll != theMTDComposition.end();
       ++icoll) {
    edm::LogPrint("SimHitCaloHitDumper") << "\n";
    edm::LogPrint("SimHitCaloHitDumper") << (*icoll).second << " hits in the event = " << (*icoll).first;
    edm::LogPrint("SimHitCaloHitDumper") << "\n";
    for (int ihit = 0; ihit < (*icoll).first; ++ihit) {
      edm::LogPrint("SimHitCaloHitDumper")
          << theMTDHits[nhit] << " Energy = " << theMTDHits[nhit].energyLoss()
          << " tid orig/offset= " << theMTDHits[nhit].originalTrackId() << " " << theMTDHits[nhit].offsetTrackId()
          << " Track Id = " << theMTDHits[nhit].trackId();
      nhit++;
    }
  }

  edm::LogPrint("SimHitCaloHitDumper") << "\n Calorimeter Hits in the event = " << theCaloHits.size();
  edm::LogPrint("SimHitCaloHitDumper") << "\n";
  //   for (std::vector<PCaloHit>::iterator isim = theCaloHits.begin();
  //        isim != theCaloHits.end(); ++isim){
  //     edm::LogPrint("SimHitCaloHitDumper") << (*isim) ;
  //   }
  nhit = 0;
  for (std::vector<std::pair<int, std::string> >::iterator icoll = theCaloComposition.begin();
       icoll != theCaloComposition.end();
       ++icoll) {
    edm::LogPrint("SimHitCaloHitDumper") << "\n";
    edm::LogPrint("SimHitCaloHitDumper") << (*icoll).second << " hits in the event = " << (*icoll).first;
    edm::LogPrint("SimHitCaloHitDumper") << "\n";
    for (int ihit = 0; ihit < (*icoll).first; ++ihit) {
      edm::LogPrint("SimHitCaloHitDumper") << theCaloHits[nhit];
      nhit++;
    }
  }

  return;
}

void SimHitCaloHitDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("TrackerHitsPixelBarrelLowTof", edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"));
  desc.add<edm::InputTag>("TrackerHitsPixelBarrelHighTof", edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelHighTof"));
  desc.add<edm::InputTag>("TrackerHitsPixelEndcapLowTof", edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"));
  desc.add<edm::InputTag>("TrackerHitsPixelEndcapHighTof", edm::InputTag("g4SimHits", "TrackerHitsPixelEndcapHighTof"));
  desc.add<edm::InputTag>("TrackerHitsTIBLowTof", edm::InputTag("g4SimHits", "TrackerHitsTIBLowTof"));
  desc.add<edm::InputTag>("TrackerHitsTIBHighTof", edm::InputTag("g4SimHits", "TrackerHitsTIBHighTof"));
  desc.add<edm::InputTag>("TrackerHitsTIDLowTof", edm::InputTag("g4SimHits", "TrackerHitsTIDLowTof"));
  desc.add<edm::InputTag>("TrackerHitsTIDHighTof", edm::InputTag("g4SimHits", "TrackerHitsTIDHighTof"));
  desc.add<edm::InputTag>("TrackerHitsTOBLowTof", edm::InputTag("g4SimHits", "TrackerHitsTOBLowTof"));
  desc.add<edm::InputTag>("TrackerHitsTOBHighTof", edm::InputTag("g4SimHits", "TrackerHitsTOBHighTof"));
  desc.add<edm::InputTag>("TrackerHitsTECLowTof", edm::InputTag("g4SimHits", "TrackerHitsTECLowTof"));
  desc.add<edm::InputTag>("TrackerHitsTECHighTof", edm::InputTag("g4SimHits", "TrackerHitsTECHighTof"));
  desc.add<edm::InputTag>("MuonDTHits", edm::InputTag("g4SimHits", "MuonDTHits"));
  desc.add<edm::InputTag>("MuonCSCHits", edm::InputTag("g4SimHits", "MuonCSCHits"));
  desc.add<edm::InputTag>("MuonRPCHits", edm::InputTag("g4SimHits", "MuonRPCHits"));
  desc.add<edm::InputTag>("FastTimerHitsBarrel", edm::InputTag("g4SimHits", "FastTimerHitsBarrel"));
  desc.add<edm::InputTag>("FastTimerHitsEndcap", edm::InputTag("g4SimHits", "FastTimerHitsEndcap"));
  desc.add<edm::InputTag>("EcalHitsEB", edm::InputTag("g4SimHits", "EcalHitsEB"));
  desc.add<edm::InputTag>("EcalHitsEE", edm::InputTag("g4SimHits", "EcalHitsEE"));
  desc.add<edm::InputTag>("EcalHitsES", edm::InputTag("g4SimHits", "EcalHitsES"));
  desc.add<edm::InputTag>("HcalHits", edm::InputTag("g4SimHits", "HcalHits"));
  desc.add<edm::InputTag>("CaloHitsTk", edm::InputTag("g4SimHits", "CaloHitsTk"));
  desc.add<edm::InputTag>("ZDCHITS", edm::InputTag("g4SimHits", "ZDCHITS"));
  desc.add<edm::InputTag>("CastorTU", edm::InputTag("g4SimHits", "CastorTU"));
  desc.add<edm::InputTag>("CastorPL", edm::InputTag("g4SimHits", "CastorPL"));
  desc.add<edm::InputTag>("CastorFI", edm::InputTag("g4SimHits", "CastorFI"));
  desc.add<edm::InputTag>("CastorBU", edm::InputTag("g4SimHits", "CastorBU"));
  descriptions.add("simHitCaloHitDumper", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(SimHitCaloHitDumper);
