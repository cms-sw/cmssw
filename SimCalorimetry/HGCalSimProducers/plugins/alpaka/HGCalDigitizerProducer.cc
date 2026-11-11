// CMSSW imports
#include "FWCore/AbstractServices/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include <iomanip>  // for std::setw
#include <future>
#include <chrono>

// CLHEP includes
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"

// Alpaka imports
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// includes for data formats
#include "DataFormats/HGCalDigi/interface/HGCalSimHitAccumHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalSimHitAccumDevice.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalDigiDevice.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalECONDPacketInfoDevice.h"
#include "CondFormats/HGCalObjects/interface/HGCalAgeingCondsHost.h"
#include "CondFormats/DataRecord/interface/HGCalAgeingCondsRcd.h"
#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"

//kernels handlers
#include "SimCalorimetry/HGCalSimProducers/interface/alpaka/HGCalDigitizerAlgorithms.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class HGCalDigitizerProducer : public stream::SynchronizingEDProducer<> {
  public:
    explicit HGCalDigitizerProducer(const edm::ParameterSet&);
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void acquire(device::Event const&, device::EventSetup const&) override;
    void produce(device::Event&, device::EventSetup const&) override;

    const edm::EDGetTokenT<hgcaldigi::HGCalSimHitAccumHost> g4HitsAccTkn_;
    edm::ESWatcher<HGCalAgeingCondsRcd> ageingCondsWatcher_;
    const edm::ESGetToken<hgcal::HGCalAgeingCondsHost, HGCalAgeingCondsRcd> ageingCondsTkn_;
    const edm::ESGetToken<hgcal::HGCalDenseIndexInfoHost, HGCalDenseIndexInfoRcd> denseIndexTkn_;
    const edm::ESGetToken<hgcal::HGCalMappingModuleParamHost, HGCalElectronicsMappingRcd> moduleInfoTkn_;
    const edm::ESGetToken<hgcal::HGCalMappingCellParamHost, HGCalElectronicsMappingRcd> cellInfoToken_;
    int n_blocks_, n_threads_;
    const HGCalAgeingAlgorithms ripener_;
    const HGCalECONDAlgorithms concentrator_;
    const HGCalROCDigitizerAlgorithms digitizer_;
    bool produceExtra_;
    std::optional<hgcaldigi::HGCalRealisticSimHitDevice> agedG4HitsAcc_device_;
    device::EDPutToken<hgcaldigi::HGCalRealisticSimHitDevice> agedG4HitsAccTkn_;
    std::optional<hgcaldigi::HGCalROCDigisAccumDevice> rocDigisAcc_device_;
    device::EDPutToken<hgcaldigi::HGCalROCDigisAccumDevice> rocDigisAccTkn_;
    std::optional<hgcaldigi::HGCalDigiDevice> rocDigis_device_;
    edm::EDPutTokenT<::hgcaldigi::HGCalDigiHost> rocDigisTkn_;
    std::optional<hgcaldigi::HGCalECONDPacketInfoDevice> econInfo_device_;
    edm::EDPutTokenT<::hgcaldigi::HGCalECONDPacketInfoHost> econInfoTkn_;
  };

  //
  HGCalDigitizerProducer::HGCalDigitizerProducer(const edm::ParameterSet& iConfig)
      : SynchronizingEDProducer(iConfig),
        g4HitsAccTkn_{consumes<hgcaldigi::HGCalSimHitAccumHost>(iConfig.getParameter<edm::InputTag>("g4HitsAcc"))},
        ageingCondsTkn_{esConsumes(iConfig.getParameter<edm::ESInputTag>("ageingConds"))},
        denseIndexTkn_{esConsumes(iConfig.getParameter<edm::ESInputTag>("denseIndexer"))},
        moduleInfoTkn_{esConsumes(iConfig.getParameter<edm::ESInputTag>("moduleInfo"))},
        cellInfoToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("cellInfo"))},
        n_blocks_{iConfig.getParameter<int>("n_blocks")},
        n_threads_{iConfig.getParameter<int>("n_threads")},
        ripener_{n_blocks_, n_threads_},
        concentrator_{n_blocks_, n_threads_},
        digitizer_{n_blocks_, n_threads_},
        produceExtra_(iConfig.getParameter<bool>("produceExtra")) {
    rocDigisTkn_ = produces("HGCROCDigis");
    econInfoTkn_ = produces("HGCECONInfo");

    //add the extra products
    if (produceExtra_) {
      agedG4HitsAccTkn_ = produces("agedG4HitsAccumulator");
      rocDigisAccTkn_ = produces("HGCROCDigisAccumulator");
    }
  }

  //
  void HGCalDigitizerProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("g4HitsAcc", edm::InputTag("hgcalDigis", "DIGI", "TEST"));
    desc.add("ageingConds", edm::ESInputTag{})->setComment("Label for ageing parameters");
    desc.add("denseIndexer", edm::ESInputTag{})->setComment("Dense indexer SoA source");
    desc.add("moduleInfo", edm::ESInputTag{})->setComment("module info SoA source");
    desc.add("cellInfo", edm::ESInputTag{})->setComment("cell info SoA source");
    desc.add<bool>("produceExtra", true)->setComment("produce extra digitizer collections");
    desc.add<int>("n_blocks", -1);
    desc.add<int>("n_threads", -1);
    descriptions.addWithDefaultLabel(desc);
  }

  //
  void HGCalDigitizerProducer::acquire(device::Event const& iEvent, device::EventSetup const& iSetup) {
    auto& queue = iEvent.queue();
    const auto& ageingConds = iSetup.getData(ageingCondsTkn_);
    const auto& denseIndexer = iSetup.getData(denseIndexTkn_);
    const auto& moduleInfo = iSetup.getData(moduleInfoTkn_);
    int nmodules = moduleInfo.view().metadata().size();
    const auto& cellInfo = iSetup.getData(cellInfoToken_);
    int ncellinfo = cellInfo.view().metadata().size();
    const auto& g4AccHits = iEvent.get(g4HitsAccTkn_);
    auto nchannels = g4AccHits.view().metadata().size();

    LogDebug("[HGCalDigitizerProducer]") << "Loaded accumulated Geant 4 hits for " << nchannels << " cells"
                                         << ", " << ageingConds.view().metadata().size() << " ageing conditions"
                                         << ", " << denseIndexer.view().metadata().size() << " dense indices"
                                         << ", " << nmodules << " modules"
                                         << ", and " << ncellinfo << " cell info indices";

#ifdef EDM_ML_DEBUG
    alpaka::wait(queue);
    auto start = std::chrono::steady_clock::now();
#endif

    //copy conditions and dense indexer to device
    hgcal::HGCalAgeingCondsDevice ageingConds_device(queue, nchannels);
    alpaka::memcpy(queue, ageingConds_device.buffer(), ageingConds.const_buffer());
    hgcal::HGCalDenseIndexInfoDevice denseIndex_device(queue, nchannels);
    alpaka::memcpy(queue, denseIndex_device.buffer(), denseIndexer.const_buffer());
    hgcal::HGCalMappingModuleParamDevice moduleInfo_device(queue, nmodules);
    alpaka::memcpy(queue, moduleInfo_device.buffer(), moduleInfo.const_buffer());
    hgcal::HGCalMappingCellParamDevice cellInfo_device(queue, ncellinfo);
    alpaka::memcpy(queue, cellInfo_device.buffer(), cellInfo.const_buffer());

    //run ageing
    agedG4HitsAcc_device_ = ripener_.age(queue, g4AccHits, ageingConds_device, denseIndex_device, cellInfo_device);

    //run digitization

    // setup RNG for noise generation
    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine& engine = rng->getEngine(iEvent.streamID());
    const uint64_t seed = static_cast<uint64_t>(CLHEP::RandFlat::shootInt(&engine, std::numeric_limits<int>::max()));

    if (agedG4HitsAcc_device_.has_value()) {
      //populate ECON-D info and common mode
      econInfo_device_ = concentrator_.init(queue, ageingConds_device, moduleInfo_device, seed);

      //digitize channels
      rocDigisAcc_device_ = digitizer_.digitize(queue,
                                                agedG4HitsAcc_device_.value(),
                                                econInfo_device_.value(),
                                                ageingConds_device,
                                                denseIndex_device,
                                                cellInfo_device,
                                                seed);

      //havest the digis
      rocDigis_device_ = digitizer_.harvest(queue, rocDigisAcc_device_.value().const_view());
    }

#ifdef EDM_ML_DEBUG
    alpaka::wait(queue);
    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<float> elapsed = stop - start;
    LogDebug("[HGCalDigitizerProducer]") << "Time spent creating digis of " << nchannels
                                         << " channels: " << elapsed.count();
#endif
  }

  //
  void HGCalDigitizerProducer::produce(device::Event& iEvent, device::EventSetup const& iSetup) {
    LogDebug("[HGCalDigitizerProducer]") << "Storing HGCAL DIGIs in the event";

    //copy to host
    auto& queue = iEvent.queue();
    auto const& digisDevice = rocDigis_device_.value();
    ::hgcaldigi::HGCalDigiHost digisHost(digisDevice.view().metadata().size());
    alpaka::memcpy(queue, digisHost.buffer(), digisDevice.const_buffer());
    auto const& econdDevice = econInfo_device_.value();
    ::hgcaldigi::HGCalECONDPacketInfoHost econdInfoHost(econdDevice.view().metadata().size());
    alpaka::memcpy(queue, econdInfoHost.buffer(), econdDevice.const_buffer());
    alpaka::wait(queue);

    //helpers
    if (produceExtra_) {
      if (agedG4HitsAcc_device_.has_value())
        iEvent.emplace(agedG4HitsAccTkn_, std::move(agedG4HitsAcc_device_.value()));

      if (rocDigisAcc_device_.has_value())
        iEvent.emplace(rocDigisAccTkn_, std::move(rocDigisAcc_device_.value()));
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalDigitizerProducer);
