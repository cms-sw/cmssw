// system include files
#include <cmath>
#include <iostream>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

namespace HGCalValidSimhitCheck {
  struct Counters {
    Counters() {
      badTypes_.clear();
      occupancy_.clear();
      goodChannels_.clear();
    }
    CMS_THREAD_GUARD(mtx_) mutable std::map<int, int> badTypes_, occupancy_;
    CMS_THREAD_GUARD(mtx_) mutable std::vector<int> goodChannels_;
    mutable std::mutex mtx_;
  };
}  // namespace HGCalValidSimhitCheck

class HGCalWaferHitCheck : public edm::stream::EDAnalyzer<edm::GlobalCache<HGCalValidSimhitCheck::Counters> > {
public:
  explicit HGCalWaferHitCheck(const edm::ParameterSet&, const HGCalValidSimhitCheck::Counters* count);

  static std::unique_ptr<HGCalValidSimhitCheck::Counters> initializeGlobalCache(edm::ParameterSet const& iConfig) {
    return std::make_unique<HGCalValidSimhitCheck::Counters>();
  }

  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endStream() override;
  static void globalEndJob(const HGCalValidSimhitCheck::Counters* counters);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  template <class T>
  void analyzeHits(const std::string&, const T&);

  // ----------member data ---------------------------
  enum inputType { Sim = 1, Digi = 2, Reco = 3 };
  const std::string nameDetector_, caloHitSource_;
  const edm::InputTag source_;
  const int inpType_;
  const int verbosity_;
  const bool ifNose_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> geomToken_;
  const edm::EDGetTokenT<HGCalDigiCollection> digiSource_;
  const edm::EDGetTokenT<HGCRecHitCollection> recHitSource_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_hit_;
  const HGCalDDDConstants* hgcons_;
  std::map<int, int> badTypes_, occupancy_;
  std::vector<int> goodChannels_;
  static const int waferMax = 12;
  static const int layerMax = 28;
};

HGCalWaferHitCheck::HGCalWaferHitCheck(const edm::ParameterSet& iConfig, const HGCalValidSimhitCheck::Counters*)
    : nameDetector_(iConfig.getParameter<std::string>("detectorName")),
      caloHitSource_(iConfig.getParameter<std::string>("caloHitSource")),
      source_(iConfig.getParameter<edm::InputTag>("source")),
      inpType_(iConfig.getParameter<int>("inputType")),
      verbosity_(iConfig.getUntrackedParameter<int>("verbosity", 0)),
      ifNose_(iConfig.getUntrackedParameter<bool>("ifNose", false)),
      geomToken_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})),
      digiSource_(consumes<HGCalDigiCollection>(source_)),
      recHitSource_(consumes<HGCRecHitCollection>(source_)),
      tok_hit_(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", caloHitSource_))) {
  edm::LogVerbatim("HGCalValidation") << "Validation for input type " << inpType_ << " for " << nameDetector_;
}

void HGCalWaferHitCheck::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detectorName", "HGCalEESensitive");
  desc.add<std::string>("caloHitSource", "HGCHitsEE");
  desc.add<edm::InputTag>("source", edm::InputTag("simHGCalUnsuppressedDigis", "EE"));
  desc.add<int>("inputType", 1);
  desc.addUntracked<int>("verbosity", 0);
  desc.addUntracked<bool>("ifNose", false);
  descriptions.add("hgcalWaferHitCheckEE", desc);
}

void HGCalWaferHitCheck::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Get collection depending on inputType
  if (inpType_ <= Sim) {
    const edm::Handle<edm::PCaloHitContainer>& theCaloHitContainer = iEvent.getHandle(tok_hit_);
    if (theCaloHitContainer.isValid()) {
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " << theCaloHitContainer->size() << " SimHits";
      analyzeHits(nameDetector_, *(theCaloHitContainer.product()));
    } else if (verbosity_ > 0) {
      edm::LogVerbatim("HGCalValidation") << "PCaloHitContainer does not exist for " << nameDetector_;
    }
  } else if (inpType_ == Digi) {
    const edm::Handle<HGCalDigiCollection>& theDigiContainer = iEvent.getHandle(digiSource_);
    if (theDigiContainer.isValid()) {
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " << theDigiContainer->size() << " Digis";
      analyzeHits(nameDetector_, *(theDigiContainer.product()));
    } else if (verbosity_ > 0) {
      edm::LogVerbatim("HGCalValidation") << "DigiContainer does not exist for " << nameDetector_;
    }
  } else {
    const edm::Handle<HGCRecHitCollection>& theRecHitContainer = iEvent.getHandle(recHitSource_);
    if (theRecHitContainer.isValid()) {
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation") << nameDetector_ << " with " << theRecHitContainer->size() << " hits";
      analyzeHits(nameDetector_, *(theRecHitContainer.product()));
    } else if (verbosity_ > 0) {
      edm::LogVerbatim("HGCalValidation") << "RecHitContainer does not exist for " << nameDetector_;
    }
  }
}

template <class T>
void HGCalWaferHitCheck::analyzeHits(std::string const& name, T const& hits) {
  for (auto const& hit : hits) {
    uint32_t id = hit.id();
    int zside, type, layer, waferU, waferV;
    if (ifNose_) {
      HFNoseDetId detId = HFNoseDetId(id);
      waferU = detId.waferU();
      waferV = detId.waferV();
      type = detId.type();
      layer = detId.layer();
      zside = detId.zside();
    } else {
      HGCSiliconDetId detId = HGCSiliconDetId(id);
      waferU = detId.waferU();
      waferV = detId.waferV();
      type = detId.type();
      layer = detId.layer();
      zside = detId.zside();
    }
    int index = HGCalWaferIndex::waferIndex(layer, waferU, waferV, false);
    int typef = hgcons_->waferType(layer, waferU, waferV, true);
    if (zside < 0)
      index *= -1;
    auto itr = occupancy_.find(index);
    if (itr == occupancy_.end())
      occupancy_[index] = 1;
    else
      ++occupancy_[index];
    if (type != typef) {
      auto ktr = badTypes_.find(index);
      if (ktr == badTypes_.end())
        badTypes_[index] = 1;
      else
        ++badTypes_[index];
      if (verbosity_ == 1)
        edm::LogVerbatim("HGCalValidation")
            << "Detector " << name << " zside = " << zside << " layer = " << layer << " type = " << type << ":" << typef
            << " wafer = " << waferU << ":" << waferV << " index " << index;
    }
    if (verbosity_ > 1)
      edm::LogVerbatim("HGCalValidation")
          << "Detector " << name << " zside = " << zside << " layer = " << layer << " type = " << type << ":" << typef
          << " wafer = " << waferU << ":" << waferV;
  }
}

// ------------ method called when starting to processes a run  ------------
void HGCalWaferHitCheck::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  hgcons_ = &iSetup.getData(geomToken_);
  if (verbosity_ > 0)
    edm::LogVerbatim("HGCalValidation") << nameDetector_ << " defined with " << hgcons_->layers(false)
                                        << " Layers with " << (hgcons_->firstLayer() - 1) << " in front";

  goodChannels_.clear();
  for (int iz = 0; iz < 2; ++iz) {
    int zside = iz * 2 - 1;
    for (int layer = 1; layer <= layerMax; ++layer) {
      for (int waferU = -waferMax; waferU <= waferMax; ++waferU) {
        for (int waferV = -waferMax; waferV <= waferMax; ++waferV) {
          int index = zside * HGCalWaferIndex::waferIndex(layer, waferU, waferV, false);
          if (hgcons_->isValidHex8(layer, waferU, waferV, true))
            goodChannels_.emplace_back(index);
        }
      }
    }
  }
}

void HGCalWaferHitCheck::endStream() {
  std::scoped_lock lock(globalCache()->mtx_);
  for (auto [id, count] : occupancy_) {
    if (globalCache()->occupancy_.find(id) == globalCache()->occupancy_.end())
      globalCache()->occupancy_[id] = count;
    else
      globalCache()->occupancy_[id] += count;
  }
  for (auto [id, count] : badTypes_) {
    if (globalCache()->badTypes_.find(id) == globalCache()->badTypes_.end())
      globalCache()->badTypes_[id] = count;
    else
      globalCache()->badTypes_[id] += count;
  }

  globalCache()->goodChannels_ = goodChannels_;
  globalCache()->mtx_.unlock();
}

void HGCalWaferHitCheck::globalEndJob(const HGCalValidSimhitCheck::Counters* count) {
  int allbad(0), nocc(0);
  for (auto const& index : count->goodChannels_) {
    int zside = (index < 0) ? -1 : 1;
    int layer = HGCalWaferIndex::waferLayer(std::abs(index));
    int waferU = HGCalWaferIndex::waferU(std::abs(index));
    int waferV = HGCalWaferIndex::waferV(std::abs(index));
    int occ = (count->occupancy_.find(index) == count->occupancy_.end()) ? 0 : count->occupancy_[index];
    int bad = (count->badTypes_.find(index) == count->badTypes_.end()) ? 0 : count->badTypes_[index];
    if (occ == 0)
      ++nocc;
    if (bad > 0)
      ++allbad;
    if (occ == 0 || bad > 0) {
      edm::LogVerbatim("HGCalValidation") << "ZS:Layer:u:v:index " << zside << ":" << layer << ":" << waferU << ":"
                                          << waferV << ":" << index << " Occ " << occ << " bad " << bad;
    }
  }
  edm::LogVerbatim("HGCalValidation") << "\n\n"
                                      << allbad << " channels with bad types among " << count->goodChannels_.size()
                                      << " channels and " << nocc << " channels with zero occupancy\n\n";
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferHitCheck);
