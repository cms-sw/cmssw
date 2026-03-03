#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cmath>

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"

class RecHitTester : public DQMEDAnalyzer {
public:
  explicit RecHitTester(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  using SimHitsT = std::vector<PCaloHit>;
  using RecHitsT = EcalRecHitCollection;
  using PFRecHitsT = reco::PFRecHitCollection;
  using UncalibRecHitsT = EcalUncalibratedRecHitCollection;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;

  const edm::EDGetTokenT<SimHitsT> ebSimHitToken_, eeSimHitToken_;
  const edm::EDGetTokenT<UncalibRecHitsT> ebUncalibRecHitToken_, eeUncalibRecHitToken_;
  const edm::EDGetTokenT<RecHitsT> ebRecHitToken_, eeRecHitToken_;
  const edm::EDGetTokenT<PFRecHitsT> PFRecHitToken_;

  std::string outFolder_;

  std::unordered_map<std::string, std::tuple<unsigned, float, float, unsigned, float, float>> histo2dVarsReco = {
      {"En_Eta", std::make_tuple(100, 0., 100., 50, -6.5, 6.5)},
      {"En_Phi", std::make_tuple(100, 0., 100., 50, -3.5, 3.5)},
      {"Eta_Phi", std::make_tuple(50, -6.5, 6.5, 50, -3.5, 3.5)},
  };

  using U2Map = std::unordered_map<std::string, MonitorElement*>;
  U2Map h2d_ebsimHits_, h2d_eesimHits_;
  U2Map h2d_ebuncalibRecHits_, h2d_eeuncalibRecHits_;
  U2Map h2d_ebrecHits_, h2d_eerecHits_;
  U2Map h2d_pfRecHits_;
};

RecHitTester::RecHitTester(const edm::ParameterSet& iConfig)
    : caloGeomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()),
      ebSimHitToken_(consumes<SimHitsT>(iConfig.getParameter<edm::InputTag>("ebSimHits"))),
      eeSimHitToken_(consumes<SimHitsT>(iConfig.getParameter<edm::InputTag>("eeSimHits"))),
      ebUncalibRecHitToken_(consumes<UncalibRecHitsT>(iConfig.getParameter<edm::InputTag>("ebUncalibRecHits"))),
      eeUncalibRecHitToken_(consumes<UncalibRecHitsT>(iConfig.getParameter<edm::InputTag>("eeUncalibRecHits"))),
      ebRecHitToken_(consumes<RecHitsT>(iConfig.getParameter<edm::InputTag>("ebRecHits"))),
      eeRecHitToken_(consumes<RecHitsT>(iConfig.getParameter<edm::InputTag>("eeRecHits"))),
      PFRecHitToken_(consumes<PFRecHitsT>(iConfig.getParameter<edm::InputTag>("pfRecHits"))),
      outFolder_(iConfig.getParameter<std::string>("outFolder")) {}

void RecHitTester::bookHistograms(DQMStore::IBooker& ibook, edm::Run const&, edm::EventSetup const&) {
  ibook.setCurrentFolder(outFolder_ + "/Hits");

  for (auto& h2dVar : histo2dVarsReco) {
    auto [nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY] = h2dVar.second;
    auto x_title = h2dVar.first.substr(0, h2dVar.first.find("_"));
    auto y_title = h2dVar.first.substr(h2dVar.first.find("_") + 1);
    h2d_ebsimHits_[h2dVar.first] = ibook.book2D(
        "EBSimHits" + h2dVar.first, "EBSimHits;" + x_title + ";" + y_title, nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY);
    h2d_eesimHits_[h2dVar.first] = ibook.book2D(
        "EESimHits" + h2dVar.first, "EESimHits;" + x_title + ";" + y_title, nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY);
    h2d_ebuncalibRecHits_[h2dVar.first] = ibook.book2D("EBUncalibRecHits" + h2dVar.first,
                                                       "EBUncalibRecHits;" + x_title + ";" + y_title,
                                                       nBinsX,
                                                       hMinX,
                                                       hMaxX,
                                                       nBinsY,
                                                       hMinY,
                                                       hMaxY);
    h2d_eeuncalibRecHits_[h2dVar.first] = ibook.book2D("EEUncalibRecHits" + h2dVar.first,
                                                       "EEUncalibRecHits;" + x_title + ";" + y_title,
                                                       nBinsX,
                                                       hMinX,
                                                       hMaxX,
                                                       nBinsY,
                                                       hMinY,
                                                       hMaxY);
    h2d_ebrecHits_[h2dVar.first] = ibook.book2D(
        "EBRecHits" + h2dVar.first, "EBRecHits;" + x_title + ";" + y_title, nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY);
    h2d_eerecHits_[h2dVar.first] = ibook.book2D(
        "EERecHits" + h2dVar.first, "EERecHits;" + x_title + ";" + y_title, nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY);
    h2d_pfRecHits_[h2dVar.first] = ibook.book2D(
        "PFRecHits" + h2dVar.first, "PFRecHits;" + x_title + ";" + y_title, nBinsX, hMinX, hMaxX, nBinsY, hMinY, hMaxY);
  }
}

void RecHitTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& caloGeom = iSetup.getData(caloGeomToken_);

  edm::Handle<SimHitsT> ebsimhitHandle;
  iEvent.getByToken(ebSimHitToken_, ebsimhitHandle);
  if (!ebsimhitHandle.isValid()) {
    edm::LogPrint("RecHitTester") << "Input EB SimHit collection not found.";
    return;
  }
  edm::Handle<SimHitsT> eesimhitHandle;
  iEvent.getByToken(eeSimHitToken_, eesimhitHandle);
  if (!eesimhitHandle.isValid()) {
    edm::LogPrint("RecHitTester") << "Input EE SimHit collection not found.";
    return;
  }

  edm::Handle<UncalibRecHitsT> ebuncalibrechitHandle;
  iEvent.getByToken(ebUncalibRecHitToken_, ebuncalibrechitHandle);
  if (!ebuncalibrechitHandle.isValid()) {
    edm::LogPrint("RecHitTester") << "Input EB UncalibRecHit collection not found.";
    return;
  }
  edm::Handle<UncalibRecHitsT> eeuncalibrechitHandle;
  iEvent.getByToken(eeUncalibRecHitToken_, eeuncalibrechitHandle);
  if (!eeuncalibrechitHandle.isValid()) {
    edm::LogPrint("RecHitTester") << "Input EE UncalibRecHit collection not found.";
    return;
  }

  edm::Handle<RecHitsT> ebrechitHandle;
  iEvent.getByToken(ebRecHitToken_, ebrechitHandle);
  if (!ebrechitHandle.isValid()) {
    edm::LogPrint("RecHitTester") << "Input EB RecHit collection not found.";
    return;
  }
  edm::Handle<RecHitsT> eerechitHandle;
  iEvent.getByToken(eeRecHitToken_, eerechitHandle);
  if (!eerechitHandle.isValid()) {
    edm::LogPrint("RecHitTester") << "Input EE RecHit collection not found.";
    return;
  }

  edm::Handle<PFRecHitsT> pfrechitHandle;
  iEvent.getByToken(PFRecHitToken_, pfrechitHandle);
  if (!pfrechitHandle.isValid()) {
    edm::LogPrint("RecHitTester") << "Input pfrechit collection not found.";
    return;
  }

  auto ebsimhits = *ebsimhitHandle;
  auto eesimhits = *eesimhitHandle;
  auto ebuncalibrechits = *ebuncalibrechitHandle;
  auto eeuncalibrechits = *eeuncalibrechitHandle;
  auto ebrechits = *ebrechitHandle;
  auto eerechits = *eerechitHandle;
  auto pfrechits = *pfrechitHandle;

  for (auto h : ebsimhits) {
    float eta = caloGeom.getPosition(h.id()).eta();
    float phi = caloGeom.getPosition(h.id()).phi();

    h2d_ebsimHits_["En_Eta"]->Fill(h.energy(), eta);
    h2d_ebsimHits_["En_Phi"]->Fill(h.energy(), phi);
    h2d_ebsimHits_["Eta_Phi"]->Fill(eta, phi);
  }
  for (auto h : eesimhits) {
    float eta = caloGeom.getPosition(h.id()).eta();
    float phi = caloGeom.getPosition(h.id()).phi();

    h2d_eesimHits_["En_Eta"]->Fill(h.energy(), eta);
    h2d_eesimHits_["En_Phi"]->Fill(h.energy(), phi);
    h2d_eesimHits_["Eta_Phi"]->Fill(eta, phi);
  }

  for (auto h : ebuncalibrechits) {
    float eta = caloGeom.getPosition(h.id()).eta();
    float phi = caloGeom.getPosition(h.id()).phi();

    h2d_ebuncalibRecHits_["En_Eta"]->Fill(h.amplitude(), eta);
    h2d_ebuncalibRecHits_["En_Phi"]->Fill(h.amplitude(), phi);
    h2d_ebuncalibRecHits_["Eta_Phi"]->Fill(eta, phi);
  }
  for (auto h : eeuncalibrechits) {
    float eta = caloGeom.getPosition(h.id()).eta();
    float phi = caloGeom.getPosition(h.id()).phi();

    h2d_eeuncalibRecHits_["En_Eta"]->Fill(h.amplitude(), eta);
    h2d_eeuncalibRecHits_["En_Phi"]->Fill(h.amplitude(), phi);
    h2d_eeuncalibRecHits_["Eta_Phi"]->Fill(eta, phi);
  }

  for (auto h : ebrechits) {
    float eta = caloGeom.getPosition(h.id()).eta();
    float phi = caloGeom.getPosition(h.id()).phi();

    h2d_ebrecHits_["En_Eta"]->Fill(h.energy(), eta);
    h2d_ebrecHits_["En_Phi"]->Fill(h.energy(), phi);
    h2d_ebrecHits_["Eta_Phi"]->Fill(eta, phi);
  }
  for (auto h : eerechits) {
    float eta = caloGeom.getPosition(h.id()).eta();
    float phi = caloGeom.getPosition(h.id()).phi();

    h2d_eerecHits_["En_Eta"]->Fill(h.energy(), eta);
    h2d_eerecHits_["En_Phi"]->Fill(h.energy(), phi);
    h2d_eerecHits_["Eta_Phi"]->Fill(eta, phi);
  }

  for (const auto& h : pfrechits) {
    float eta = caloGeom.getPosition(h.detId()).eta();
    float phi = caloGeom.getPosition(h.detId()).phi();

    h2d_pfRecHits_["En_Eta"]->Fill(h.energy(), eta);
    h2d_pfRecHits_["En_Phi"]->Fill(h.energy(), phi);
    h2d_pfRecHits_["Eta_Phi"]->Fill(eta, phi);
  }
}

void RecHitTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("outFolder", "HLT/ParticleFlow");
  desc.add<edm::InputTag>("ebSimHits", edm::InputTag("g4SimHits", "EcalHitsEB"));
  desc.add<edm::InputTag>("eeSimHits", edm::InputTag("g4SimHits", "EcalHitsEE"));
  desc.add<edm::InputTag>("ebUncalibRecHits", edm::InputTag("hltEcalUncalibRecHit", "EcalUncalibRecHitsEE"));
  desc.add<edm::InputTag>("eeUncalibRecHits", edm::InputTag("hltEcalUncalibRecHit", "EcalUncalibRecHitsEE"));
  desc.add<edm::InputTag>("ebRecHits", edm::InputTag("hltEcalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("eeRecHits", edm::InputTag("hltEcalRecHit", "EcalRecHitsEE"));
  desc.add<edm::InputTag>("pfRecHits", edm::InputTag("hltParticleFlowRecHitECALUnseeded"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(RecHitTester);
