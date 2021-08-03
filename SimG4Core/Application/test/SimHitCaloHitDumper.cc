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

SimHitCaloHitDumper::SimHitCaloHitDumper(const edm::ParameterSet& iConfig)
    : moduleName(iConfig.getParameter<std::string>("moduleLabelG4")) {
  PixelBarrelLowTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsPixelBarrelLowTof"));
  PixelBarrelHighTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsPixelBarrelHighTof"));
  PixelEndcapLowTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsPixelEndcapLowTof"));
  PixelEndcapHighTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsPixelEndcapHighTof"));
  TrackerTIBLowTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTIBLowTof"));
  TrackerTIBHighTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTIBHighTof"));
  TrackerTIDLowTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTIDLowTof"));
  TrackerTIDHighTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTIDHighTof"));
  TrackerTOBLowTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTOBLowTof"));
  TrackerTOBHighTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTOBHighTof"));
  TrackerTECLowTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTECLowTof"));
  TrackerTECHighTofToken_ =
      consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "TrackerHitsTECHighTof"));

  MuonDTToken_ = consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "MuonDTHits"));
  MuonCSCToken_ = consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "MuonCSCHits"));
  MuonRPCToken_ = consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "MuonRPCHits"));

  FastTimerBTLToken_ = consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "FastTimerHitsBarrel"));
  FastTimerETLToken_ = consumes<edm::PSimHitContainer>(edm::InputTag(std::string(moduleName), "FastTimerHitsEndcap"));

  EcalEBToken_ = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "EBHits"));
  EcalEEToken_ = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "EEHits"));
  EcalESToken_ = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "ESHits"));
  HcalToken_ = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "HcalHits"));
  CaloTkToken_ = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "CaloTkHits"));
  ZDCToken_ = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "ZDC"));
  CastorTUToken_ = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "CastorTUHits"));
  CastorPLToken_ = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "CastorPLHits"));
  CastorFIToken_ = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "CastorFIHits"));
  CastorBUToken_ = consumes<edm::PCaloHitContainer>(edm::InputTag(std::string(moduleName), "CastorBUHits"));
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

  iEvent.getByToken(PixelBarrelLowTofToken_, PixelBarrelHitsLowTof);
  iEvent.getByToken(PixelBarrelHighTofToken_, PixelBarrelHitsHighTof);
  iEvent.getByToken(PixelEndcapLowTofToken_, PixelEndcapHitsLowTof);
  iEvent.getByToken(PixelEndcapHighTofToken_, PixelEndcapHitsHighTof);
  iEvent.getByToken(TrackerTIBLowTofToken_, TIBHitsLowTof);
  iEvent.getByToken(TrackerTIBHighTofToken_, TIBHitsHighTof);
  iEvent.getByToken(TrackerTIDLowTofToken_, TIDHitsLowTof);
  iEvent.getByToken(TrackerTIDHighTofToken_, TIDHitsHighTof);
  iEvent.getByToken(TrackerTOBLowTofToken_, TOBHitsLowTof);
  iEvent.getByToken(TrackerTOBHighTofToken_, TOBHitsHighTof);
  iEvent.getByToken(TrackerTECLowTofToken_, TECHitsLowTof);
  iEvent.getByToken(TrackerTECHighTofToken_, TECHitsHighTof);

  iEvent.getByToken(MuonDTToken_, DTHits);
  iEvent.getByToken(MuonCSCToken_, CSCHits);
  iEvent.getByToken(MuonRPCToken_, RPCHits);

  iEvent.getByToken(EcalEBToken_, EBHits);
  iEvent.getByToken(EcalEEToken_, EEHits);
  iEvent.getByToken(EcalESToken_, ESHits);
  iEvent.getByToken(HcalToken_, HcalHits);
  iEvent.getByToken(CaloTkToken_, CaloTkHits);
  iEvent.getByToken(ZDCToken_, ZDCHits);
  iEvent.getByToken(CastorTUToken_, CastorTUHits);
  iEvent.getByToken(CastorPLToken_, CastorPLHits);
  iEvent.getByToken(CastorFIToken_, CastorFIHits);
  iEvent.getByToken(CastorBUToken_, CastorBUHits);

  iEvent.getByToken(FastTimerBTLToken_, BTLHits);
  iEvent.getByToken(FastTimerETLToken_, ETLHits);

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
      edm::LogPrint("SimHitCaloHitDumper") << theMTDHits[nhit] << " Track Id = " << theMTDHits[nhit].trackId();
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
  desc.add<std::string>("moduleLabelG4", "g4SimHits")->setComment("Module for input SimHit/CaloHit collections");
  descriptions.add("simHitCaloHitDumper", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
DEFINE_FWK_MODULE(SimHitCaloHitDumper);
