#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <iomanip>
#include <map>

#include <TFormula.h>
#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/OTRecHitsSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/StubsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/StackedModuleGeometryRecord.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAGeometrySoACollection.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BLMaterialMapCollection.h"
#include "RecoTracker/Record/interface/BLMaterialMapRecord.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/StackedModuleGeometryHost.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/StackedModuleGeometrySoACollection.h"
#include "CAHitNtupletGenerator.h"

#include "HeterogeneousCore/AlpakaCore/interface/MoveToDeviceCache.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

// #define GPU_DEBUG

namespace reco {
  struct CAGeometryParams {
    // The layer- and layer-pair dependent settings come from the `geometry` PSet, the generic cut
    // scalars from the top level of the module PSet. This struct holds them in that form and the CA
    // geometry product is built from it below, converting the two per-LAYER triplet cuts into the
    // per-LAYER-PAIR columns the SoA carries (see the broadcast in globalBeginRun).
    //
    // The members whose name ends in PerPair, plus floorDCACuts_ and the three maxStub* vectors, are
    // the OT-stub extension: they are optional and empty when the configuration does not set them,
    // in which case the per-layer / scalar value is broadcast instead (or the cut is disabled).
    CAGeometryParams(edm::ParameterSet const& iConfig)
        : CAGeometryParams(iConfig, iConfig.getParameterSet("geometry")) {}

    CAGeometryParams(edm::ParameterSet const& iConfig, edm::ParameterSet const& geo)
        :  // ---- graph (per layer pair) ----
          layerPairs_(geo.getParameter<std::vector<unsigned int>>("pairGraph")),
          startingPairs_(geo.getParameter<std::vector<unsigned int>>("startingPairs")),
          skipsLayers_(geo.getParameter<std::vector<unsigned int>>("skipsLayers")),
          // ---- doublet cuts (per layer pair) ----
          phiCuts_(geo.getParameter<std::vector<int>>("phiCuts")),
          minInner_(geo.getParameter<std::vector<double>>("minInner")),
          maxInner_(geo.getParameter<std::vector<double>>("maxInner")),
          minOuter_(geo.getParameter<std::vector<double>>("minOuter")),
          maxOuter_(geo.getParameter<std::vector<double>>("maxOuter")),
          maxDR_(geo.getParameter<std::vector<double>>("maxDR")),
          minDZ_(geo.getParameter<std::vector<double>>("minDZ")),
          maxDZ_(geo.getParameter<std::vector<double>>("maxDZ")),
          ptCuts_(geo.getParameter<std::vector<double>>("ptCuts")),
          // ---- doublet cuts (scalars, top level) ----
          cellZ0Cut_(iConfig.getParameter<double>("cellZ0Cut")),
          dzdrFact_(iConfig.getParameter<double>("dzdrFact")),
          minYsizeB1_(iConfig.getParameter<int>("minYsizeB1")),
          minYsizeB2_(iConfig.getParameter<int>("minYsizeB2")),
          maxDYsize12_(iConfig.getParameter<int>("maxDYsize12")),
          maxDYsize_(iConfig.getParameter<int>("maxDYsize")),
          maxDYPred_(iConfig.getParameter<int>("maxDYPred")),
          // ---- triplet cuts (per layer) ----
          caDCACuts_(geo.getParameter<std::vector<double>>("caDCACuts")),
          caThetaCuts_(geo.getParameter<std::vector<double>>("caThetaCuts")),
          // ---- triplet cuts (scalars, top level) ----
          ptmin_(iConfig.getParameter<double>("ptmin")),
          hardCurvCut_(iConfig.getParameter<double>("hardCurvCut")),
          maxPhiResid_(iConfig.getParameter<double>("maxPhiResid")),
          sameDPhiSign_(iConfig.getParameter<bool>("sameDPhiSign")),
          // ---- ntuplet cuts + fishbone (per layer) ----
          startMaxInnerR_(geo.getParameter<std::vector<double>>("startMaxInnerR")),
          maxDCurv_(geo.getParameter<std::vector<double>>("maxDCurv")),
          floorDCurv_(geo.getParameter<std::vector<double>>("floorDCurv")),
          fishboneCuts_(geo.getParameter<std::vector<double>>("fishboneCuts")),
          // ---- OT-stub extension (all optional, per layer pair) ----
          caDCACutsPerPair_(optionalVDouble(geo, "caDCACutsPerPair")),
          caThetaCutsPerPair_(optionalVDouble(geo, "caThetaCutsPerPair")),
          cellZ0CutPerPair_(optionalVDouble(geo, "cellZ0CutPerPair")),
          floorDCACuts_(optionalVDouble(geo, "floorDCACuts")),
          maxStubCurvSigma_(optionalVDouble(geo, "maxStubCurvSigma")),
          maxStubGeomCurvSigma_(optionalVDouble(geo, "maxStubGeomCurvSigma")),
          maxStubInnerDoubletDCurv_(optionalVDouble(geo, "maxStubInnerDoubletDCurv")) {
      // Is there a starting pair whose inner layer is not BPix1?
      startNoBPix1_ = false;
      for (const unsigned int& i : startingPairs_) {
        if (layerPairs_[2 * i] > 0) {
          startNoBPix1_ = true;
          break;
        }
      }

      // Size validation. Everything per layer pair must agree with the graph, everything per layer
      // must agree with the layer count, and every starting-pair id must address an existing pair.
      const size_t nPairs = layerPairs_.size() / 2;
      const size_t nLayers = caThetaCuts_.size();
      auto checkPairs = [&](std::vector<double> const& v, char const* name) {
        if (!v.empty() && v.size() != nPairs)
          throw cms::Exception("Configuration") << "CAHitNtupletAlpaka: geometry." << name << " has " << v.size()
                                                << " entries but the CA graph has " << nPairs << " layer pairs.";
      };
      if (layerPairs_.size() % 2 != 0)
        throw cms::Exception("Configuration")
            << "CAHitNtupletAlpaka: geometry.pairGraph has an odd number of entries (" << layerPairs_.size() << ").";
      if (skipsLayers_.size() != nPairs)
        throw cms::Exception("Configuration") << "CAHitNtupletAlpaka: geometry.skipsLayers has " << skipsLayers_.size()
                                              << " entries but the CA graph has " << nPairs << " layer pairs.";
      if (phiCuts_.size() != nPairs)
        throw cms::Exception("Configuration") << "CAHitNtupletAlpaka: geometry.phiCuts has " << phiCuts_.size()
                                              << " entries but the CA graph has " << nPairs << " layer pairs.";
      for (auto const* p : {&minInner_, &maxInner_, &minOuter_, &maxOuter_, &maxDR_, &minDZ_, &maxDZ_, &ptCuts_})
        if (p->size() != nPairs)
          throw cms::Exception("Configuration")
              << "CAHitNtupletAlpaka: a per-layer-pair vector of the geometry PSet has " << p->size()
              << " entries but the CA graph has " << nPairs << " layer pairs.";
      for (auto const* p : {&caDCACuts_, &startMaxInnerR_, &maxDCurv_, &floorDCurv_, &fishboneCuts_})
        if (p->size() != nLayers)
          throw cms::Exception("Configuration")
              << "CAHitNtupletAlpaka: a per-layer vector of the geometry PSet has " << p->size()
              << " entries but caThetaCuts declares " << nLayers << " layers.";
      checkPairs(caDCACutsPerPair_, "caDCACutsPerPair");
      checkPairs(caThetaCutsPerPair_, "caThetaCutsPerPair");
      checkPairs(cellZ0CutPerPair_, "cellZ0CutPerPair");
      checkPairs(floorDCACuts_, "floorDCACuts");
      checkPairs(maxStubCurvSigma_, "maxStubCurvSigma");
      checkPairs(maxStubGeomCurvSigma_, "maxStubGeomCurvSigma");
      checkPairs(maxStubInnerDoubletDCurv_, "maxStubInnerDoubletDCurv");
      for (const unsigned int& i : startingPairs_)
        if (i >= nPairs)
          throw cms::Exception("Configuration") << "CAHitNtupletAlpaka: geometry.startingPairs contains the pair id "
                                                << i << " but the CA graph only has " << nPairs << " layer pairs.";
      for (size_t i = 0; i < layerPairs_.size(); ++i)
        if (layerPairs_[i] >= nLayers)
          throw cms::Exception("Configuration")
              << "CAHitNtupletAlpaka: geometry.pairGraph refers to the layer " << layerPairs_[i]
              << " but caThetaCuts only declares " << nLayers << " layers.";

#ifdef GPU_DEBUG
      std::cout << "\n========== CAGeometryParams CONSTRUCTOR ==========" << std::endl;
      std::cout << "Reading geometry from Python ParameterSet..." << std::endl;
      std::cout << "  caThetaCuts size: " << caThetaCuts_.size() << std::endl;
      std::cout << "  caDCACuts size: " << caDCACuts_.size() << std::endl;
      std::cout << "  pairGraph size: " << layerPairs_.size() << " (= " << nPairs << " pairs)" << std::endl;
      std::cout << "  startingPairs number: " << startingPairs_.size() << std::endl;
      std::cout << "  phiCuts size: " << phiCuts_.size() << std::endl;
      std::cout << "  First 5 phiCuts values: ";
      for (size_t i = 0; i < std::min(size_t(5), phiCuts_.size()); ++i) {
        std::cout << phiCuts_[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "==================================================\n" << std::endl;
#endif
    }

    // Reads an optional vdouble of the geometry PSet, returning an empty vector when it is not set.
    static std::vector<double> optionalVDouble(edm::ParameterSet const& pset, char const* name) {
      return pset.existsAs<std::vector<double>>(name) ? pset.getParameter<std::vector<double>>(name)
                                                      : std::vector<double>{};
    }

    // graph (per layer pair)
    const std::vector<unsigned int> layerPairs_;
    const std::vector<unsigned int> startingPairs_;
    const std::vector<unsigned int> skipsLayers_;

    // doublet cuts (per layer pair)
    const std::vector<int> phiCuts_;
    const std::vector<double> minInner_;
    const std::vector<double> maxInner_;
    const std::vector<double> minOuter_;
    const std::vector<double> maxOuter_;
    const std::vector<double> maxDR_;
    const std::vector<double> minDZ_;
    const std::vector<double> maxDZ_;
    const std::vector<double> ptCuts_;

    // doublet cuts (scalars)
    const double cellZ0Cut_;
    const double dzdrFact_;
    const int minYsizeB1_;
    const int minYsizeB2_;
    const int maxDYsize12_;
    const int maxDYsize_;
    const int maxDYPred_;

    // triplet cuts (per layer)
    const std::vector<double> caDCACuts_;
    const std::vector<double> caThetaCuts_;

    // triplet cuts (scalars)
    const double ptmin_;
    const double hardCurvCut_;
    const double maxPhiResid_;
    const bool sameDPhiSign_;

    // ntuplet cuts and fishbone (per layer)
    const std::vector<double> startMaxInnerR_;
    const std::vector<double> maxDCurv_;
    const std::vector<double> floorDCurv_;
    const std::vector<double> fishboneCuts_;

    // OT-stub extension (per layer pair, empty when not configured)
    const std::vector<double> caDCACutsPerPair_;
    const std::vector<double> caThetaCutsPerPair_;
    const std::vector<double> cellZ0CutPerPair_;
    const std::vector<double> floorDCACuts_;
    const std::vector<double> maxStubCurvSigma_;
    const std::vector<double> maxStubGeomCurvSigma_;
    const std::vector<double> maxStubInnerDoubletDCurv_;

    bool startNoBPix1_;

    mutable edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tokenGeometry_;
    mutable edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tokenTopology_;
    mutable edm::ESGetToken<::reco::StackedModuleGeometryHost, StackedModuleGeometryRecord> tokenStackedGeometry_;
  };

}  // namespace reco

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Per-run device-resident CA geometry.
  struct CARunGeometry {
    using CAGeometryCache = cms::alpakatools::MoveToDeviceCache<Device, ::reco::CAGeometryHost>;
    CARunGeometry(::reco::CAGeometryHost&& geometry) : geometry_(std::move(geometry)) {}
    CAGeometryCache geometry_;
  };

  template <typename TrackerTraits>
  class CAHitNtupletAlpaka
      : public stream::SynchronizingEDProducer<edm::GlobalCache<::reco::CAGeometryParams>, edm::RunCache<CARunGeometry>> {
    using HitsConstView = ::reco::TrackingRecHitConstView;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;

    using HitsOnDeviceRefProdVector = edm::RefProdVector<HitsOnDevice>;

    using TkSoAHost = ::reco::TracksHost;
    using TkSoADevice = reco::TracksSoACollection;

    using Algo = CAHitNtupletGenerator<TrackerTraits>;

    using CAGeometryCache = cms::alpakatools::MoveToDeviceCache<Device, ::reco::CAGeometryHost>;
    using Rotation = SOARotation<float>;
    using Frame = SOAFrame<float>;

  public:
    explicit CAHitNtupletAlpaka(const edm::ParameterSet& iConfig, const ::reco::CAGeometryParams* iCache);
    ~CAHitNtupletAlpaka() override = default;

    // acquire() launches the whole CA build (kernels + the one async offsets readback, see
    // CAHitNtupletGenerator::beginTuplesAsync); the framework schedules produce only after
    // this event's queue has drained -- the wait happens in the framework's async callback,
    // never on a TBB thread. produce() launches the fit passes and classification (consuming
    // the landed offsets waitlessly) and puts the product.
    void acquire(device::Event const& iEvent, device::EventSetup const& es) override;
    void produce(device::Event& iEvent, device::EventSetup const& es) override;

    // Always-on overflow surfacing: the generator accumulates capped-container overflow counts
    // per stream (doStats-independent); report once at stream end. The module label is passed so
    // several CA instances of the same job are distinguishable in the message.
    void endStream() override { deviceAlgo_.reportOverflows(moduleDescription().moduleLabel()); }

    static void globalEndJob(::reco::CAGeometryParams const*) { /* Do nothing */ };
    static void globalEndRun(edm::Run const& iRun,
                             edm::EventSetup const&,
                             RunContext const* iContext) { /* Do nothing */ };

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    static std::shared_ptr<CARunGeometry> globalBeginRun(edm::Run const& iRun,
                                                         edm::EventSetup const& iSetup,
                                                         GlobalCache const* iCache) {
      // The size consistency of the whole geometry PSet is validated (with explicit exceptions) in
      // the CAGeometryParams constructor; the asserts below only restate the two invariants this
      // function relies on directly.
      int n_layers = iCache->caThetaCuts_.size();
      int n_pairs = iCache->layerPairs_.size() / 2;
      int n_modules = 0;

#ifdef GPU_DEBUG
      std::cout << "No. Layers to be used = " << n_layers << std::endl;
      std::cout << "No. Pairs to be used = " << n_pairs << std::endl;
#endif

      assert(int(n_pairs) == int(iCache->maxDR_.size()));
      assert(int(*std::max_element(iCache->layerPairs_.begin(), iCache->layerPairs_.end())) < n_layers);
      assert(iCache->startingPairs_.empty() ||
             int(*std::max_element(iCache->startingPairs_.begin(), iCache->startingPairs_.end())) < n_pairs);

      auto const& trackerGeometry = iSetup.getData(iCache->tokenGeometry_);
      auto const& trackerTopology = iSetup.getData(iCache->tokenTopology_);
      auto const& dets = trackerGeometry.dets();

      // Get stacked module geometry for Phase-2 OT with stubs
      ::reco::StackedModuleGeometryHost const* stackedGeometry = nullptr;
      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        stackedGeometry = &iSetup.getData(iCache->tokenStackedGeometry_);
      }

#ifdef GPU_DEBUG
      auto subSystem = 0;
      auto subSystemName = GeomDetEnumerators::tkDetEnum[subSystem];
      std::cout
          << "========================================================================================================="
          << std::endl;
#endif

      auto oldLayer = 0u;
      auto layerCount = 0u;

      std::vector<bool> layerIsBarrel(n_layers);
      std::vector<bool> layerIsOT(n_layers);
      std::vector<bool> layerIsSS(n_layers);
      std::vector<int> layerStarts(n_layers + 1);
      //^ why n_layers + 1? This is a cumulative sum of the number
      // of modules each layer has. And we need the  extra spot
      // at the end to hold the total number of modules.

      std::vector<int> moduleToindexInDets;

      auto isPinPSinOTBarrel = [&](DetId detId) {
        // Select only P-hits from the OT barrel
        return (trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP &&
                detId.subdetId() == StripSubdetector::TOB);
      };
      auto isPixel = [&](DetId detId) {
        auto subId = detId.subdetId();
        return (subId == PixelSubdetector::PixelBarrel || subId == PixelSubdetector::PixelEndcap);
      };
      auto isBarrel = [&](DetId detId) {
        auto subId = detId.subdetId();
        auto subDetector = trackerGeometry.geomDetSubDetector(subId);
        return GeomDetEnumerators::isBarrel(subDetector);
      };

      // loop over all detector modules and build the CA layers
      int counter = 0;
      for (auto& det : dets) {
        DetId detid = det->geographicalId();
        auto layer = trackerTopology.layer(detid);
        // Logic:
        // - if we are not inside pixels, we need to ignore anything **but** the OT.
        // - for the time being, this is assuming that the CA extension will
        //   only cover the OT barrel part, and will ignore the OT forward.

#ifdef GPU_DEBUG
        auto subId = detid.subdetId();
        if (subSystemName != trackerGeometry.geomDetSubDetector(subId)) {
          subSystemName = trackerGeometry.geomDetSubDetector(subId);
          std::cout << " ===================== Subsystem: " << subSystemName << std::endl;
        }
#endif

        // Modules of the pixel layers
        if (isPixel(detid)) {
          if (layer != oldLayer) {
#ifdef GPU_DEBUG
            std::cout << "Pixel LayerStart: CA layer " << layerCount << " at subdetector layer " << layer
                      << " starts at module " << n_modules << " and is " << (isBarrel(detid) ? "barrel" : "not barrel")
                      << std::endl;
#endif
            layerIsBarrel[layerCount] = isBarrel(detid);
            layerIsOT[layerCount] = false;  // we are in the loop over pixel dets, so these are not OT layers
            layerIsSS[layerCount] = false;  // we are in the loop over pixel dets, so these are not SS layers
            layerStarts[layerCount++] = n_modules;
            if (layerCount >= layerStarts.size())
              break;
            oldLayer = layer;
          }
          moduleToindexInDets.push_back(counter);
          n_modules++;
        }

        // if we are using the CA extension for Phase-2,
        // we also have to collect the modules from the considered OT layers
        if constexpr (std::is_same_v<pixelTopology::Phase2OT, TrackerTraits>) {
          auto const& detUnits = det->components();
          for (auto& detUnit : detUnits) {
            DetId unitDetId(detUnit->geographicalId());
            // Modules of the considered OT layers
            if (isPinPSinOTBarrel(unitDetId)) {
              if (layer != oldLayer) {
#ifdef GPU_DEBUG
                std::cout << "OT LayerStart: CA layer " << layerCount << " at subdetector layer " << layer
                          << " starts at module " << n_modules << " and is "
                          << (isBarrel(detid) ? "barrel" : "not barrel") << std::endl;
#endif
                layerIsBarrel[layerCount] = isBarrel(detid);
                layerIsOT[layerCount] = true;   // we are in the loop over PS modules, so these are all OT layers
                layerIsSS[layerCount] = false;  // we are in the loop over PS modules, so these are not SS layers
                layerStarts[layerCount++] = n_modules;
                if (layerCount >= layerStarts.size())
                  break;
                oldLayer = layer;
              }
              moduleToindexInDets.push_back(counter);
              n_modules++;
            }
          }
        }
        counter++;
      }

      // Process OT stacked modules for Phase-2 with stubs
      // CA layers follow inside-out ordering:
      // - CA layers 28-33: Barrel (layers 1-6)
      // - CA layers 34-43: Backward disks (layers 1-5) split in two groups:
      //                    34-38 with PS modules first, then SS; 39-43 with no PS/SS split
      // - CA layers 44-53: Forward disks (layers 1-5) split in two groups:
      //                    44-48 with PS modules first, then SS; 49-53 with no PS/SS split
      //
      // IMPORTANT: StackedModuleGeometry is ALREADY sorted in CA order by StackedModuleGeometryESProducer
      // (barrel by layer -> backward by layer -> forward by layer) using stable_sort.
      // We iterate through it in index order WITHOUT re-sorting to ensure the frame array
      // index matches the detectorIndex assigned to hits/stubs (detectorIndex = nPixelModules + geomIndex).
      // Number of pixel modules already processed; OT modules start at this CA module index.
      // Used below when populating CAModulesSoA::innerSensorFrame for OT modules.
      const int nPixelModulesInCA = n_modules;

      if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
        if (stackedGeometry != nullptr) {
          auto stackedView = stackedGeometry->view();
          const uint32_t nStackedModules = static_cast<uint32_t>(stackedView.metadata().size());

          bool firstModule = true;
          uint8_t prevLayer = 0;
          bool prevIsPS = false;
          int prevCategory = -1;  // 0=barrel, 1=backward, 2=forward

          // Iterate through StackedModuleGeometry in index order (already sorted in CA order)
          for (uint32_t i = 0; i < nStackedModules; ++i) {
            bool isBarrel = stackedView.isBarrel()[i];
            bool isFwdEndcap = stackedView.isFwdEndcap()[i];
            bool isPS = stackedView.isPS()[i];
            uint8_t otLayer = stackedView.layer()[i];
            DetId stackedDetId(stackedView.stackedDetId()[i]);

            // Determine category: 0=barrel, 1=backward, 2=forward
            int category = isBarrel ? 0 : (isFwdEndcap ? 2 : 1);

            // Check if we've transitioned to a new CA layer
            // A new layer starts when category changes OR layer number changes within same category
            if (firstModule || category != prevCategory || otLayer != prevLayer || isPS != prevIsPS) {
              // Start new CA layer
              if (layerCount < layerStarts.size()) {
                layerIsBarrel[layerCount] = isBarrel;
                layerIsOT[layerCount] =
                    true;  // we are in the loop over StackedModuleGeometry, so these are all OT layers
                layerIsSS[layerCount] = !isPS;
                layerStarts[layerCount++] = n_modules;

#ifdef GPU_DEBUG
                const char* categoryName = isBarrel ? "barrel" : (isFwdEndcap ? "forward" : "backward");
                std::cout << "OT LayerStart: CA layer " << (layerCount - 1) << " starts at module " << n_modules << " ("
                          << categoryName << " layer " << int(otLayer) << ")" << std::endl;
#endif
              }
              prevCategory = category;
              prevLayer = otLayer;
              prevIsPS = isPS;
              firstModule = false;
            }

            // Find this module in TrackerGeometry dets list
            bool found = false;
            for (int detIdx = 0; detIdx < static_cast<int>(dets.size()); ++detIdx) {
              if (dets[detIdx]->geographicalId() == stackedDetId) {
                moduleToindexInDets.push_back(detIdx);
                n_modules++;
                found = true;
                break;
              }
            }
            if (!found) {
              edm::LogWarning("CAHitNtuplet")
                  << "Could not find stacked module " << stackedDetId.rawId() << " in TrackerGeometry";
            }
          }
        }
      }

#ifdef GPU_DEBUG
      std::cout << "Full CA LayerStart: " << n_layers << " layers with " << n_modules << " modules in total."
                << std::endl;
#endif
      layerStarts[n_layers] = n_modules;

      reco::CAGeometryHost product{
          cms::alpakatools::host(), n_layers + 1, n_pairs, n_pairs, n_pairs, n_layers, n_modules};

      auto layerSoA = product.view().layers();
      auto graphSoA = product.view().graph();
      auto doubletCutsSoA = product.view().doubletCuts();
      auto tripletCutsSoA = product.view().tripletCuts();
      auto ntupletCutsSoA = product.view().ntupletCuts();
      auto modulesSoA = product.view().modules();

      // For Phase2OTStubs, stackedView is needed below to pick the right per-sensor
      // frame for each OT module's `innerSensorFrame` per the contract:
      //   - PSP (moduleType==0): inner = lower sensor (P-side).
      //   - PSS (moduleType==1): inner = upper sensor (P-side).
      //   - SS:                   inner = physically-inner = lower if !isFlipped, else upper.
      auto pickInnerFrame = [&](int iCA) -> Frame {
        // Default fallback: identity-handling via detFrame for non-OT or missing stacked info.
        if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
          if (stackedGeometry != nullptr && iCA >= nPixelModulesInCA) {
            auto stackedView = stackedGeometry->view();
            const uint32_t nStackedModules = static_cast<uint32_t>(stackedView.metadata().size());
            const uint32_t iGeom = static_cast<uint32_t>(iCA - nPixelModulesInCA);
            if (iGeom < nStackedModules) {
              uint8_t mt = stackedView.moduleType()[iGeom];
              bool isFlipped = stackedView.isFlipped()[iGeom];
              bool useUpper;
              if (mt == 0)
                useUpper = false;  // PSP: pixel is lower
              else if (mt == 1)
                useUpper = true;  // PSS: pixel is upper
              else
                useUpper = isFlipped;  // SS: inner = lower if !flipped, else upper
              return useUpper ? Frame(stackedView[iGeom].upperSensorFrame())
                              : Frame(stackedView[iGeom].lowerSensorFrame());
            }
          }
        }
        // Pixel modules (or non-OTStubs topology): innerSensorFrame == detFrame.
        return modulesSoA[iCA].detFrame();
      };

      for (int i = 0; i < n_modules; ++i) {
        auto idx = moduleToindexInDets[i];
        auto det = dets[idx];
        auto vv = det->surface().position();
        auto rr = Rotation(det->surface().rotation());
        modulesSoA[i].detFrame() = Frame(vv.x(), vv.y(), vv.z(), rr);
        modulesSoA[i].innerSensorFrame() = pickInnerFrame(i);
#ifdef GPU_DEBUG
        auto const& detUnits = det->components();
        for (auto& detUnit : detUnits) {
          DetId unitDetId(detUnit->geographicalId());
          if (isPinPSinOTBarrel(unitDetId)) {
            std::cout << "Filling frame at index " << idx << " in SoA position " << i << " for det "
                      << det->geographicalId() << " and detUnit->index: " << detUnit->index() << std::endl;
          }
        }
        std::cout << "Filling frame at index " << idx << " in SoA position " << i << " for det "
                  << det->geographicalId() << std::endl;
        std::cout << "Position: " << vv << " with Rotation: " << det->surface().rotation() << std::endl;
        std::cout << "Rotation in z-r plane: "
                  << atan2(det->surface().normalVector().perp(), det->surface().normalVector().z()) * 180. / M_PI
                  << std::endl;
#endif
      }

      for (int i = 0; i < n_layers; ++i) {
        layerSoA.fishboneCut()[i] = iCache->fishboneCuts_[i];
        layerSoA.layerStarts()[i] = layerStarts[i];
        layerSoA.isBarrel()[i] = layerIsBarrel[i];
        layerSoA.isOT()[i] = layerIsOT[i];
        layerSoA.isSS()[i] = layerIsSS[i];
        ntupletCutsSoA.startMaxInnerR()[i] = iCache->startMaxInnerR_[i];
        ntupletCutsSoA.maxDCurv()[i] = iCache->maxDCurv_[i];
        ntupletCutsSoA.floorDCurv()[i] = iCache->floorDCurv_[i];
      }

      layerSoA.layerStarts()[n_layers] = layerStarts[n_layers];

      // Per-layer-pair fill. The two triplet cuts the configuration expresses PER LAYER are broadcast
      // onto the pairs here, each pair taking the value of its OWN INNER layer. That is exactly the
      // indexing the kernels read back:
      //   * TripletCuts::accept reads maxDCA/floorDCA at the INNER cell's pair (L1,L2), whose inner
      //     layer is the triplet's innermost layer L1  ->  caDCACuts[L1], upstream's anchor;
      //   * it reads maxRZTolerance at the OUTER cell's pair (L2,L3), whose inner layer is the
      //     triplet's middle layer L2                  ->  caThetaCuts[L2], upstream's anchor.
      // The scalar cellZ0Cut is broadcast unchanged onto every pair. The "...PerPair" vectors of the
      // OT-stub extension replace the corresponding broadcast when they are configured.
      for (int i = 0; i < n_pairs; ++i) {
        const uint32_t innerLayer = iCache->layerPairs_[2 * i];
        graphSoA.layerPair()[i] = {{uint32_t(iCache->layerPairs_[2 * i]), uint32_t(iCache->layerPairs_[2 * i + 1])}};
        graphSoA.skipsLayers()[i] = uint16_t(bool(iCache->skipsLayers_[i]));
        graphSoA.startingPair()[i] = false;  // set from the id list below
        doubletCutsSoA.maxDPhi()[i] = iCache->phiCuts_[i];
        doubletCutsSoA.minInner()[i] = iCache->minInner_[i];
        doubletCutsSoA.maxInner()[i] = iCache->maxInner_[i];
        doubletCutsSoA.minOuter()[i] = iCache->minOuter_[i];
        doubletCutsSoA.maxOuter()[i] = iCache->maxOuter_[i];
        doubletCutsSoA.maxDZ()[i] = iCache->maxDZ_[i];
        doubletCutsSoA.minDZ()[i] = iCache->minDZ_[i];
        doubletCutsSoA.maxDR()[i] = iCache->maxDR_[i];
        // convert ptCut in curvature radius in cm
        // 1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field
        const float minRadius = iCache->ptCuts_[i] * 87.78f;
        // Use minRadius^2/4 in the CA to avoid sqrt
        const float minRadius2T4 = 4.f * minRadius * minRadius;
        doubletCutsSoA.minPt()[i] = minRadius2T4;
        doubletCutsSoA.maxZ0()[i] =
            (!iCache->cellZ0CutPerPair_.empty()) ? iCache->cellZ0CutPerPair_[i] : iCache->cellZ0Cut_;
        tripletCutsSoA.maxRZTolerance()[i] =
            (!iCache->caThetaCutsPerPair_.empty()) ? iCache->caThetaCutsPerPair_[i] : iCache->caThetaCuts_[innerLayer];
        tripletCutsSoA.maxDCA()[i] =
            (!iCache->caDCACutsPerPair_.empty()) ? iCache->caDCACutsPerPair_[i] : iCache->caDCACuts_[innerLayer];
        // OT-stub extension cuts: from the configuration when present, otherwise the disabled
        // sentinel -1.
        tripletCutsSoA.floorDCA()[i] =
            (!iCache->floorDCACuts_.empty()) ? static_cast<float>(iCache->floorDCACuts_[i]) : -1.0f;
        doubletCutsSoA.maxStubCurvSigma()[i] =
            (!iCache->maxStubCurvSigma_.empty()) ? static_cast<float>(iCache->maxStubCurvSigma_[i]) : -1.0f;
        tripletCutsSoA.maxStubGeomCurvSigma()[i] =
            (!iCache->maxStubGeomCurvSigma_.empty()) ? static_cast<float>(iCache->maxStubGeomCurvSigma_[i]) : -1.0f;
        tripletCutsSoA.maxStubInnerDoubletDCurv()[i] = (!iCache->maxStubInnerDoubletDCurv_.empty())
                                                           ? static_cast<float>(iCache->maxStubInnerDoubletDCurv_[i])
                                                           : -1.0f;
      }

      // upstream's starting-pair id list
      for (const unsigned int& i : iCache->startingPairs_)
        graphSoA.startingPair()[i] = true;

      doubletCutsSoA.dzdrFact() = iCache->dzdrFact_;
      doubletCutsSoA.minInnerSizeB1() = iCache->minYsizeB1_;
      doubletCutsSoA.minInnerSizeB2() = iCache->minYsizeB2_;
      doubletCutsSoA.maxDSizeB1() = iCache->maxDYsize12_;
      doubletCutsSoA.maxDSize() = iCache->maxDYsize_;
      doubletCutsSoA.maxDSizePred() = iCache->maxDYPred_;

      tripletCutsSoA.ptmin() = iCache->ptmin_;
      tripletCutsSoA.maxCurv() = iCache->hardCurvCut_;
      tripletCutsSoA.maxPhiResid() = iCache->maxPhiResid_;
      tripletCutsSoA.sameDPhiSign() = iCache->sameDPhiSign_;

#ifdef GPU_DEBUG
      // Debug output: Print geometry values from Python config
      std::cout << "\n========== CA GEOMETRY FROM PYTHON CONFIG ==========" << std::endl;
      std::cout << "Number of layers: " << n_layers << std::endl;
      std::cout << "Number of layer pairs: " << n_pairs << std::endl;

      std::cout << "\n--- Layer Pair Geometry (all pairs) ---" << std::endl;
      std::cout << "Pair | Inner | Outer | phiCut | minIn | maxIn | minOut | maxOut | maxDR | minDZ | maxDZ | minPt | "
                   "stubSigma | start"
                << std::endl;
      std::cout << "-----|-------|-------|--------|-------|-------|--------|--------|-------|-------|-------|-------|--"
                   "---------|------"
                << std::endl;
      for (int i = 0; i < n_pairs; ++i) {
        float stubSig = (!iCache->maxStubCurvSigma_.empty()) ? iCache->maxStubCurvSigma_[i] : -1.0;
        std::cout << std::setw(4) << i << " | " << std::setw(5) << iCache->layerPairs_[2 * i] << " | " << std::setw(5)
                  << iCache->layerPairs_[2 * i + 1] << " | " << std::setw(6) << iCache->phiCuts_[i] << " | "
                  << std::setw(5) << iCache->minInner_[i] << " | " << std::setw(5) << iCache->maxInner_[i] << " | "
                  << std::setw(6) << iCache->minOuter_[i] << " | " << std::setw(6) << iCache->maxOuter_[i] << " | "
                  << std::setw(5) << iCache->maxDR_[i] << " | " << std::setw(5) << iCache->minDZ_[i] << " | "
                  << std::setw(5) << iCache->maxDZ_[i] << " | " << std::setw(5) << iCache->ptCuts_[i] << " | "
                  << std::setw(9) << stubSig << " | " << (graphSoA.startingPair()[i] ? "Y" : "N") << std::endl;
      }

      std::cout << "\n--- Layer Cuts (all layers) ---" << std::endl;
      std::cout << "Layer | isBarrel | caThetaCut | caDCACut | startMaxInnerR | fishboneCut" << std::endl;
      std::cout << "------|----------|------------|----------|----------------|------------" << std::endl;
      // caThetaCuts/caDCACuts are the PER-LAYER configuration vectors (size n_layers), the form the
      // geometry PSet declares; the SoA columns they are broadcast onto are per layer PAIR.
      for (int i = 0; i < n_layers; ++i) {
        std::cout << std::setw(5) << i << " | " << std::setw(8) << (layerIsBarrel[i] ? "Y" : "N") << " | "
                  << std::setw(10) << iCache->caThetaCuts_[i] << " | " << std::setw(8) << iCache->caDCACuts_[i] << " | "
                  << std::setw(14) << iCache->startMaxInnerR_[i] << " | " << std::setw(11) << iCache->fishboneCuts_[i]
                  << std::endl;
      }
      std::cout << "====================================================\n" << std::endl;
#endif

      return std::make_shared<CARunGeometry>(std::move(product));
    }

    static std::unique_ptr<::reco::CAGeometryParams> initializeGlobalCache(edm::ParameterSet const& iConfig) {
      return std::make_unique<::reco::CAGeometryParams>(iConfig);
    }

  private:
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
    // The BL-fit material map. The fit reads it only under useFitCorrections, so
    // it is consumed only then: the other topologies (Run 3 pixel tracks in particular) run in menus
    // that do not provide this record.
    bool useFitCorrections_ = false;
    device::ESGetToken<BLMaterialMap, BLMaterialMapRecord> tokenBLMaterialMap_;
    const device::EDGetToken<HitsOnDevice> pixelRecHitToken_;
    device::EDGetToken<HitsOnDevice> trackerRecHitToken_;
    const device::EDPutToken<TkSoADevice> tokenTrack_;

    const ::reco::FormulaEvaluator maxNumberOfDoublets_;
    const ::reco::FormulaEvaluator maxNumberOfTuples_;
    const uint32_t minNumberOfDoublets_;
    const uint32_t minNumberOfTuples_;

    Algo deviceAlgo_;

    bool usePhase2_ = false;

    // Seam state (per stream): the pending CA build acquire hands to produce, and the
    // no-BPix1 early-out flag (no CA ran; produce emplaces an empty collection).
    std::optional<typename Algo::PendingTuples> pending_;
    bool noBPix1_ = false;
  };

  template <typename TrackerTraits>
  CAHitNtupletAlpaka<TrackerTraits>::CAHitNtupletAlpaka(const edm::ParameterSet& iConfig,
                                                        const ::reco::CAGeometryParams* iCache)
      : SynchronizingEDProducer(iConfig),
        tokenField_(esConsumes()),
        pixelRecHitToken_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        tokenTrack_(produces()),
        maxNumberOfDoublets_(iConfig.getParameter<std::string>("maxNumberOfDoublets")),
        maxNumberOfTuples_(iConfig.getParameter<std::string>("maxNumberOfTuples")),
        minNumberOfDoublets_(iConfig.getParameter<uint32_t>("minNumberOfDoublets")),
        minNumberOfTuples_(iConfig.getParameter<uint32_t>("minNumberOfTuples")),
        deviceAlgo_(iConfig) {
    auto trackerRecHitsSoAInputTag = iConfig.getParameter<edm::InputTag>("trackerRecHitsSoA");
    if (!trackerRecHitsSoAInputTag.label().empty()) {
      trackerRecHitToken_ = consumes(trackerRecHitsSoAInputTag);
      usePhase2_ = true;
    }
    useFitCorrections_ = iConfig.getParameter<bool>("useFitCorrections");
    if (useFitCorrections_) {
      tokenBLMaterialMap_ = esConsumes();
    }
    iCache->tokenGeometry_ = esConsumes<edm::Transition::BeginRun>();
    iCache->tokenTopology_ = esConsumes<edm::Transition::BeginRun>();
    if constexpr (std::is_same_v<pixelTopology::Phase2OTStubs, TrackerTraits>) {
      iCache->tokenStackedGeometry_ = esConsumes<edm::Transition::BeginRun>();
    }
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
    desc.add<edm::InputTag>("trackerRecHitsSoA", edm::InputTag(""));

    Algo::fillPSetDescription(desc);
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::acquire(device::Event const& iEvent, device::EventSetup const& es) {
    // Release any state left from a previous event whose produce() was skipped, so its buffers are
    // destroyed here (deterministically, before new work is enqueued) rather than at an arbitrary
    // later point when their producing queue may already be recycled.
    pending_.reset();
    auto bf = 1. / es.getData(tokenField_).inverseBzAtOriginInGeV();
    // BL-fit Geant4 material map, resident on the device as an EventSetup portable condition (copied once
    // per IOV). On the serial/CPU backend data() points at the host buffer.
    const float* rhoMapDevice = useFitCorrections_ ? es.getData(tokenBLMaterialMap_).data() : nullptr;

    auto const& geometry = runCache()->geometry_.get(iEvent.queue());
    const auto& pixColl = iEvent.get(pixelRecHitToken_);

    HitsOnDeviceRefProdVector hitsCollections;
    hitsCollections.push_back(edm::RefProd<HitsOnDevice>(&pixColl));

    if (usePhase2_) {
      const auto& trkColl = iEvent.get(trackerRecHitToken_);
      hitsCollections.push_back(edm::RefProd<HitsOnDevice>(&trkColl));
    }

    uint32_t nHits = 0;
    for (auto const& ref : hitsCollections)
      nHits += ref->nHits();

    const int32_t offsetBPIX2 = hitsCollections[0]->offsetBPIX2();

    /// Don't bother if no hits on BPix1 and no good graph for that
    /// (so no staring pair without BPix1 as first layer).
    /// TODO: this could be extended to a more general check for
    /// no hits on any of the starting layers.
    noBPix1_ = !(globalCache()->startNoBPix1_ or offsetBPIX2 > 0);
    if (noBPix1_) {
      pending_.reset();
      return;  // produce emplaces the empty collection
    }

    std::array<double, 1> nHitsV{static_cast<double>(nHits)};
    std::array<double, 1> emptyV;

    // Floor the hit-dependent buffer sizes. The formulas are fitted to collision occupancy,
    // where the doublet count grows quadratically (doublets are pairs); extrapolated
    // downwards they undershoot the real demand of a quiet event, where nearly every doublet
    // is a genuine track segment instead of combinatorial junk. The floors are configuration
    // parameters sized to the demand of quiet events with a margin; the remaining tail is
    // truncated and counted by the always-on overflow sentinel. A floor of zero would also
    // collapse the capacity-bound launch block-counts to an invalid 0-block configuration.
    uint32_t const maxTuples = std::max<uint32_t>(maxNumberOfTuples_.evaluate(nHitsV, emptyV), minNumberOfTuples_);
    uint32_t const maxDoublets =
        std::max<uint32_t>(maxNumberOfDoublets_.evaluate(nHitsV, emptyV), minNumberOfDoublets_);

#ifdef CA_PIPELINE_COUNTERS
    printf("[CA Pipeline] Event: run=%u lumi=%u event=%llu nHits=%u\n",
           iEvent.id().run(),
           iEvent.id().luminosityBlock(),
           (unsigned long long)iEvent.id().event(),
           nHits);
#endif

    // The whole CA build (hit prep, doublets, connect, ntuplets) + one async D2H of the
    // tuple-multiplicity offsets. No blocking wait anywhere: the framework's seam runs
    // produce only after this queue has drained.
    pending_ = deviceAlgo_.beginTuplesAsync(
        hitsCollections, geometry, bf, maxDoublets, maxTuples, iEvent.queue(), rhoMapDevice);
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::produce(device::Event& iEvent, device::EventSetup const& es) {
    if (noBPix1_) {
      edm::LogWarning("CAHitNtupletAlpaka") << "No hit on BPix1 and all the starting pairs have BPix1 as inner "
                                            << "layer.\nIt's useless to run the CA. Returning with 0 tracks!";
      auto& queue = iEvent.queue();
      reco::TracksSoACollection tracks(queue, 0, 0);
      auto ntracks_d = cms::alpakatools::make_device_view(queue, tracks.view().tracks().nTracks());
      alpaka::memset(queue, ntracks_d, 0);
      iEvent.emplace(tokenTrack_, std::move(tracks));
      return;
    }

    // Fit passes + classification, consuming the offsets that landed across the seam;
    // zero readbacks, zero waits.
    auto const& geometry = runCache()->geometry_.get(iEvent.queue());
    const auto& pixColl = iEvent.get(pixelRecHitToken_);

    HitsOnDeviceRefProdVector hitsCollections;
    hitsCollections.push_back(edm::RefProd<HitsOnDevice>(&pixColl));

    if (usePhase2_) {
      const auto& trkColl = iEvent.get(trackerRecHitToken_);
      hitsCollections.push_back(edm::RefProd<HitsOnDevice>(&trkColl));
    }

    iEvent.emplace(tokenTrack_,
                   deviceAlgo_.finishTuplesAsync(std::move(*pending_), hitsCollections, geometry, iEvent.queue()));
    pending_.reset();
  }

  using CAHitNtupletAlpakaPhase1 = CAHitNtupletAlpaka<pixelTopology::Phase1>;
  using CAHitNtupletAlpakaHIonPhase1 = CAHitNtupletAlpaka<pixelTopology::HIonPhase1>;
  using CAHitNtupletAlpakaPhase2 = CAHitNtupletAlpaka<pixelTopology::Phase2>;
  using CAHitNtupletAlpakaPhase2OT = CAHitNtupletAlpaka<pixelTopology::Phase2OT>;
  using CAHitNtupletAlpakaPhase2OTStubs = CAHitNtupletAlpaka<pixelTopology::Phase2OTStubs>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaHIonPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2OT);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2OTStubs);
