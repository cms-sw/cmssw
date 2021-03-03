// system include files
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

class HcalDumpHits : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HcalDumpHits(const edm::ParameterSet&);
  ~HcalDumpHits() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override {}
  void endJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  const edm::EDGetTokenT<edm::PCaloHitContainer> simHitSource_;
  const edm::EDGetTokenT<QIE11DigiCollection> hbheDigiSource_;
  const edm::EDGetTokenT<QIE10DigiCollection> hfDigiSource_;
  const edm::EDGetTokenT<HODigiCollection> hoDigiSource_;
  const edm::EDGetTokenT<HBHERecHitCollection> hbheRecHitSource_;
  const edm::EDGetTokenT<HFRecHitCollection> hfRecHitSource_;
  const edm::EDGetTokenT<HORecHitCollection> hoRecHitSource_;
  const edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_HRNDC_;
  const HcalDDDRecConstants* hcons_;
};

HcalDumpHits::HcalDumpHits(const edm::ParameterSet& iConfig)
    : simHitSource_(consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("simHitSource"))),
      hbheDigiSource_(consumes<QIE11DigiCollection>(iConfig.getParameter<edm::InputTag>("hbheDigiSource"))),
      hfDigiSource_(consumes<QIE10DigiCollection>(iConfig.getParameter<edm::InputTag>("hfDigiSource"))),
      hoDigiSource_(consumes<HODigiCollection>(iConfig.getParameter<edm::InputTag>("hoDigiSource"))),
      hbheRecHitSource_(consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheRecHitSource"))),
      hfRecHitSource_(consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfR(ecHitSource"))),
      hoRecHitSource_(consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoRecHitSource"))),
      tok_HRNDC_(esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord, edm::Transition::BeginRun>()),
      hcons_(nullptr) {}

void HcalDumpHits::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("simHitSource", edm::InputTag("g4SimHits", "HcalHits"));
  desc.add<edm::InputTag>("hbheDigiSource", edm::InputTag("hcalDigis"));
  desc.add<edm::InputTag>("hfDigiSource", edm::InputTag("hcalDigis"));
  desc.add<edm::InputTag>("hoDigiSource", edm::InputTag("hcalDigis"));
  desc.add<edm::InputTag>("hbheRecHitSource", edm::InputTag("hbhereco"));
  desc.add<edm::InputTag>("hfRecHitSource", edm::InputTag("hfreco"));
  desc.add<edm::InputTag>("hoRecHitSource", edm::InputTag("horeco"));
  descriptions.add("hcalDumpHits", desc);
}

void HcalDumpHits::beginRun(const edm::Run&, const edm::EventSetup& iSetup) { hcons_ = &iSetup.getData(tok_HRNDC_); }

void HcalDumpHits::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  // first SimHits
  edm::Handle<edm::PCaloHitContainer> theCaloHitContainer;
  iEvent.getByToken(simHitSource_, theCaloHitContainer);
  if (theCaloHitContainer.isValid()) {
    edm::LogVerbatim("HcalValidation") << theCaloHitContainer->size() << " SimHits in HCAL";
    unsigned int k(0);
    for (auto const& hit : *(theCaloHitContainer.product())) {
      unsigned int id = hit.id();
      HcalDetId hid = HcalHitRelabeller::relabel(id, hcons_);
      edm::LogVerbatim("HcalValidation") << "[" << k << "] " << hid << " E " << hit.energy() << " T " << hit.time();
      ++k;
    }
  }

  // Digis (HBHE/HF/HO)
  edm::Handle<QIE11DigiCollection> hbheDigiCollection;
  iEvent.getByToken(hbheDigiSource_, hbheDigiCollection);
  if (hbheDigiCollection.isValid()) {
    edm::LogVerbatim("HcalValidation") << hbheDigiCollection->size() << " Digis for HB/HE";
    unsigned int k(0);
    for (auto const& it : *(hbheDigiCollection.product())) {
      QIE11DataFrame hit(it);
      std::ostringstream ost;
      ost << "[" << k << "] " << HcalDetId(hit.detid()) << " with " << hit.size() << " words:";
      unsigned int k1(0);
      for (auto itr = hit.begin(); itr != hit.end(); ++itr, ++k1)
        ost << " [" << k1 << "] " << (*itr);
      edm::LogVerbatim("HcalValidation") << ost.str();
      ++k;
    }
  }
  edm::Handle<QIE10DigiCollection> hfDigiCollection;
  iEvent.getByToken(hfDigiSource_, hfDigiCollection);
  if (hfDigiCollection.isValid()) {
    edm::LogVerbatim("HcalValidation") << hfDigiCollection->size() << " Digis for HF";
    unsigned int k(0);
    for (auto const& it : *(hfDigiCollection.product())) {
      QIE10DataFrame hit(it);
      std::ostringstream ost;
      ost << "[" << k << "] " << HcalDetId(hit.detid()) << " with " << hit.size() << " words ";
      unsigned int k1(0);
      for (auto itr = hit.begin(); itr != hit.end(); ++itr, ++k1)
        ost << " [" << k1 << "] " << (*itr);
      edm::LogVerbatim("HcalValidation") << ost.str();
      ++k;
    }
  }
  edm::Handle<HODigiCollection> hoDigiCollection;
  iEvent.getByToken(hoDigiSource_, hoDigiCollection);
  if (hoDigiCollection.isValid()) {
    edm::LogVerbatim("HcalValidation") << hoDigiCollection->size() << " Digis for HO";
    unsigned int k(0);
    for (auto const& it : *(hoDigiCollection.product())) {
      HODataFrame hit(it);
      std::ostringstream ost;
      ost << "[" << k << "] " << HcalDetId(hit.id()) << " with " << hit.size() << " words ";
      for (int k1 = 0; k1 < hit.size(); ++k1)
        ost << " [" << k1 << "] " << hit.sample(k1);
      edm::LogVerbatim("HcalValidation") << ost.str();
      ++k;
    }
  }

  // RecHits (HBHE/HF/HO)
  edm::Handle<HBHERecHitCollection> hbhecoll;
  iEvent.getByToken(hbheRecHitSource_, hbhecoll);
  if (hbhecoll.isValid()) {
    edm::LogVerbatim("HcalValidation") << hbhecoll->size() << " RecHits for HB/HE";
    unsigned int k(0);
    for (const auto& it : *(hbhecoll.product())) {
      HBHERecHit hit(it);
      edm::LogVerbatim("HcalValidation") << "[" << k << "] = " << hit;
      ++k;
    }
  }
  edm::Handle<HFRecHitCollection> hfcoll;
  iEvent.getByToken(hfRecHitSource_, hfcoll);
  if (hfcoll.isValid()) {
    edm::LogVerbatim("HcalValidation") << hfcoll->size() << " RecHits for HF";
    unsigned int k(0);
    for (const auto& it : *(hfcoll.product())) {
      HFRecHit hit(it);
      edm::LogVerbatim("HcalValidation") << "[" << k << "] = " << hit;
      ++k;
    }
  }
  edm::Handle<HORecHitCollection> hocoll;
  iEvent.getByToken(hoRecHitSource_, hocoll);
  if (hocoll.isValid()) {
    edm::LogVerbatim("HcalValidation") << hocoll->size() << " RecHits for HO";
    unsigned int k(0);
    for (const auto& it : *(hocoll.product())) {
      HORecHit hit(it);
      edm::LogVerbatim("HcalValidation") << "[" << k << "] = " << hit;
      ++k;
    }
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_FWK_MODULE(HcalDumpHits);
