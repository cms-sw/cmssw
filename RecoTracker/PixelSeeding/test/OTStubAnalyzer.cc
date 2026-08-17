#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"

#include <set>
#include <sstream>
#include <bitset>

class OTStubAnalyzer : public edm::stream::EDAnalyzer<> {
public:
  explicit OTStubAnalyzer(edm::ParameterSet const& iConfig);
  ~OTStubAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<reco::OTRecHitsHost> recHitsToken_;
  edm::EDGetTokenT<reco::StubsHost> bendStubToken_;
  edm::EDGetTokenT<reco::StubsHost> vectorHitStubToken_;

  edm::ESGetToken<::reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> geomToken_;

  int maxHitsToPrint_;
  int maxStubsToPrint_;
  bool printRecHits_;
  bool printBendStubs_;
  bool printVectorHitStubs_;
};

OTStubAnalyzer::OTStubAnalyzer(edm::ParameterSet const& iConfig)
    : recHitsToken_(consumes(iConfig.getParameter<edm::InputTag>("recHits"))),
      bendStubToken_(consumes(iConfig.getParameter<edm::InputTag>("bendStubs"))),
      vectorHitStubToken_(consumes(iConfig.getParameter<edm::InputTag>("vectorHitStubs"))),
      geomToken_(esConsumes()),
      maxHitsToPrint_(iConfig.getParameter<int>("maxHitsToPrint")),
      maxStubsToPrint_(iConfig.getParameter<int>("maxStubsToPrint")),
      printRecHits_(iConfig.getParameter<bool>("printRecHits")),
      printBendStubs_(iConfig.getParameter<bool>("printBendStubs")),
      printVectorHitStubs_(iConfig.getParameter<bool>("printVectorHitStubs")) {}

void OTStubAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recHits", edm::InputTag("pixelSeedingOTRecHitsSoAConverter"))
      ->setComment("Input OT RecHits SoA collection");
  desc.add<edm::InputTag>("bendStubs", edm::InputTag("otStubProducer"))
      ->setComment("Input bend-based stubs collection");
  desc.add<edm::InputTag>("vectorHitStubs", edm::InputTag("otStubProducerVectorHitStyle"))
      ->setComment("Input VectorHits-style stubs collection");
  desc.add<int>("maxHitsToPrint", 20)->setComment("Maximum number of hits to print");
  desc.add<int>("maxStubsToPrint", 10)->setComment("Maximum number of stubs to print per collection");
  desc.add<bool>("printRecHits", false)->setComment("Print OT RecHits");
  desc.add<bool>("printBendStubs", false)->setComment("Print bend-based stubs");
  desc.add<bool>("printVectorHitStubs", false)->setComment("Print VectorHits-style stubs");
  descriptions.addWithDefaultLabel(desc);
}

void OTStubAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  if (printRecHits_) {
    auto const& recHitsHost = iEvent.get(recHitsToken_);
    auto const& geomHost = iSetup.getData(geomToken_);

    auto const& hitsView = recHitsHost.const_view().otRecHits();
    auto const& moduleView = recHitsHost.const_view().otHitModules();
    auto const& geomView = geomHost.const_view();
    uint32_t nHits = hitsView.metadata().size();
    uint32_t nModules = recHitsHost.nModules();
    uint32_t nGeomModules = geomView.metadata().size();

    edm::LogPrint("OTStubAnalyzer") << "\n=== OT RecHits ===";
    edm::LogPrint("OTStubAnalyzer") << "Total number of hits: " << nHits;
    edm::LogPrint("OTStubAnalyzer") << "Number of modules: " << nModules;
    edm::LogPrint("OTStubAnalyzer") << "Number of geometry modules: " << nGeomModules;

    // OTRecHitsSoA has no isLower column: a hit is on the lower sensor when its SoA index is
    // below the stack's upperSensorStart. detIdx = nPixelModules + iGeom (converter convention;
    // nPixelModules is 4000 in Phase-2).
    constexpr uint32_t kNPixelModules = 4000;
    uint32_t nToPrint = std::min(nHits, static_cast<uint32_t>(maxHitsToPrint_));
    for (uint32_t i = 0; i < nToPrint; ++i) {
      uint32_t detIdx = hitsView[i].detectorIndex();
      uint32_t iGeom = (detIdx >= kNPixelModules) ? (detIdx - kNPixelModules) : 0;
      bool isLower = (iGeom < static_cast<uint32_t>(moduleView.metadata().size()))
                         ? (i < moduleView[iGeom].upperSensorStart())
                         : true;
      bool isFlipped = (iGeom < nGeomModules) ? geomView[iGeom].isFlipped() : false;
      // If not flipped, lower=inner; if flipped, lower=outer.
      bool isInner = isFlipped ? !isLower : isLower;

      float xg = hitsView[i].xGlobal();
      float yg = hitsView[i].yGlobal();
      std::ostringstream msg;
      msg << "Hit " << i << ":"
          << " globalPos=(" << xg << ", " << yg << ", " << hitsView[i].zGlobal() << ")"
          << " r=" << std::sqrt(xg * xg + yg * yg) << " iphi=" << unsafe_atan2s<7>(yg, xg) << " localPos=("
          << hitsView[i].xLocal() << ", " << hitsView[i].yLocal() << ")"
          << " localErr=(" << hitsView[i].xerrLocal() << ", " << hitsView[i].yerrLocal() << ")"
          << " detIdx=" << detIdx << " stackDetId="
          << ((iGeom < static_cast<uint32_t>(moduleView.metadata().size())) ? moduleView[iGeom].stackDetId() : 0u)
          << " sensorDetId=" << hitsView[i].sensorDetId() << " isLower=" << isLower
          << " (physical: " << (isInner ? "inner" : "outer") << ")";
      edm::LogPrint("OTStubAnalyzer") << msg.str();
    }

    if (nHits > nToPrint) {
      edm::LogPrint("OTStubAnalyzer") << "... (" << (nHits - nToPrint) << " more hits not shown)";
    }

    edm::LogPrint("OTStubAnalyzer") << "\n=== Module Start Indices (for modules with hits) ===";
    std::set<uint32_t> uniqueDetIndices;
    for (uint32_t i = 0; i < nHits; ++i) {
      uniqueDetIndices.insert(hitsView[i].detectorIndex());
    }

    for (auto detIdx : uniqueDetIndices) {
      if (detIdx < nModules) {
        edm::LogPrint("OTStubAnalyzer") << "Module " << detIdx << ": moduleStart=" << moduleView[detIdx].moduleStart()
                                        << " upperSensorStart=" << moduleView[detIdx].upperSensorStart();
      }
    }

    edm::LogPrint("OTStubAnalyzer") << "\n=== Module Start Indices (first 10 for reference) ===";
    uint32_t nModulesToPrint = std::min(nModules + 1, 10u);
    for (uint32_t i = 0; i < nModulesToPrint; ++i) {
      edm::LogPrint("OTStubAnalyzer") << "Module " << i << ": moduleStart=" << moduleView[i].moduleStart()
                                      << " upperSensorStart=" << moduleView[i].upperSensorStart();
    }
  }

  if (printBendStubs_) {
    auto const& bendStubsHost = iEvent.get(bendStubToken_);

    auto const& stubsView = bendStubsHost.const_view().stubs();
    uint32_t nStubs = stubsView.metadata().size();

    edm::LogPrint("OTStubAnalyzer") << "\n=== Bend-Based Stubs ===";
    edm::LogPrint("OTStubAnalyzer") << "Total number of stubs: " << nStubs;

    uint32_t nToPrint = std::min(nStubs, static_cast<uint32_t>(maxStubsToPrint_));
    for (uint32_t i = 0; i < nToPrint; ++i) {
      std::ostringstream msg;
      msg << std::fixed << std::setprecision(4);  // in cm, means microns

      msg << "Stub " << std::setw(2) << std::right << i << ": "

          << " dPhiDr=" << std::setw(8) << stubsView[i].dPhiDr() << " dPhiDrErr=" << std::setw(8)
          << stubsView[i].dPhiDrError() << " "

          << " innerHit=" << std::setw(6) << stubsView[i].lowerHitIdx() << " outerHit=" << std::setw(6)
          << stubsView[i].upperHitIdx();

      uint8_t flags = stubsView[i].flags();
      bool isBarrel = reco::StubFlags::isBarrel(flags);
      bool isFlat = reco::StubFlags::isFlat(flags);
      uint8_t layer = reco::StubFlags::layer(flags);
      msg << (isBarrel ? "B" : "E") << static_cast<int>(layer);
      if (isBarrel)
        msg << (isFlat ? "_flat" : "_tilt");

      edm::LogPrint("OTStubAnalyzer") << msg.str();
    }

    if (nStubs > nToPrint) {
      edm::LogPrint("OTStubAnalyzer") << "... (" << (nStubs - nToPrint) << " more stubs not shown)";
    }
  }

  if (printVectorHitStubs_) {
    auto const& vectorHitStubsHost = iEvent.get(vectorHitStubToken_);

    auto const& stubsView = vectorHitStubsHost.const_view().stubs();
    uint32_t nStubs = stubsView.metadata().size();

    edm::LogPrint("OTStubAnalyzer") << "\n=== VectorHits-Style Stubs ===";
    edm::LogPrint("OTStubAnalyzer") << "Total number of stubs: " << nStubs;

    uint32_t nToPrint = std::min(nStubs, static_cast<uint32_t>(maxStubsToPrint_));
    for (uint32_t i = 0; i < nToPrint; ++i) {
      std::ostringstream msg;

      msg << std::fixed << std::setprecision(4);  // in cm, means microns

      msg << "Stub " << std::setw(2) << std::right << i << ": "

          << " dPhiDr=" << std::setw(8) << stubsView[i].dPhiDr() << " dPhiDrErr=" << std::setw(8)
          << stubsView[i].dPhiDrError() << " "

          << " innerHit=" << std::setw(6) << stubsView[i].lowerHitIdx() << " outerHit=" << std::setw(6)
          << stubsView[i].upperHitIdx();

      uint8_t flags = stubsView[i].flags();
      bool isBarrel = reco::StubFlags::isBarrel(flags);
      bool isFlat = reco::StubFlags::isFlat(flags);
      uint8_t layer = reco::StubFlags::layer(flags);
      msg << (isBarrel ? "B" : "E") << static_cast<int>(layer);
      if (isBarrel)
        msg << (isFlat ? "_flat" : "_tilt");

      edm::LogPrint("OTStubAnalyzer") << msg.str();
    }

    if (nStubs > nToPrint) {
      edm::LogPrint("OTStubAnalyzer") << "... (" << (nStubs - nToPrint) << " more stubs not shown)";
    }
  }

  edm::LogPrint("OTStubAnalyzer") << "";
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OTStubAnalyzer);
