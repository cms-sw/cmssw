#include <iostream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <cstdint>

// framework imports
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

// alpaka imports
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

// data formats
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/HGCalDigi/interface/HGCalSimHitAccumHost.h"
#include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDevice.h"
#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

typedef std::vector<edm::ParameterSet> VPSet;

/**
  @short a helper structure to hold a PCaloHit source that needs to be processed and how it should be handled
  it also contains the internal mapping of det id to global logical dense index
 */
struct PCaloHitConfig {
  PCaloHitConfig(int32_t d, bool isc, double deltaT, edm::InputTag tag, edm::EDGetTokenT<edm::PCaloHitContainer> t)
      : det(d), iscalib(isc), tof_delay(deltaT), tkn(t) {
    edm::LogWarning("HGCalSimHitAccumulator")
        << "will consume PCaloHits collection for det=" << det << " TOF delay=" << tof_delay << " iscalib=" << iscalib
        << " collection is" << tag.label() << ":" << tag.instance() << ":" << tag.process();
  }
  int32_t det;
  bool iscalib;
  double tof_delay;
  edm::EDGetTokenT<edm::PCaloHitContainer> tkn;
  mutable std::unordered_map<uint32_t, int32_t> det2dense;  //allowed to be updated
};

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  class HGCalSimHitAccumulator : public global::EDProducer<> {
  public:
    explicit HGCalSimHitAccumulator(const edm::ParameterSet&);

    /**
    @short declares configuration parameters allowed
    */
    static void fillDescriptions(edm::ConfigurationDescriptions&);

    /**
    @short the actual producing method
    */
    void produce(edm::StreamID, device::Event&, device::EventSetup const&) const override;

  private:
    /**
      @short looks up the appropriate PCaloHitConfig to use
    */
    std::vector<PCaloHitConfig>::const_iterator findHitConsumerConfig(int32_t det, bool iscalib) const;

    /**
    @short reads the electronics dense indexing and builds an internal DetId -> DenseIndex mapping
    The mapping is only built once in a run. The `ncellsMapped_` variable is used to track if the map is available or not
    */
    int32_t buildGeoToDenseIndexMap(const hgcal::HGCalDenseIndexInfoDevice&,
                                    const hgcal::HGCalMappingCellParamHost&) const;

    const bool produceExtra_;
    const device::ESGetToken<hgcal::HGCalDenseIndexInfoDevice, HGCalDenseIndexInfoRcd> indexingToken_;
    const edm::ESGetToken<hgcal::HGCalMappingCellParamHost, HGCalElectronicsMappingRcd> cellMappingToken_;
    const edm::EDPutTokenT<hgcaldigi::HGCalSimHitAccumHost> g4AccumulatorToken_;
    edm::EDPutTokenT<hgcaldigi::HGCalSimHitAccumulatorMonitorCollection> accMonitorToken_;
    std::vector<PCaloHitConfig> hitConsumerConfig_;
    mutable std::once_flag mapInitFlag_;
    mutable int32_t nMaxReadoutCells_ = 0;
    static constexpr float nsPerBx_ = 25.0f;
  };

  //
  HGCalSimHitAccumulator::HGCalSimHitAccumulator(const edm::ParameterSet& iConfig)
      : EDProducer<>(iConfig),
        produceExtra_(iConfig.getParameter<bool>("produceExtra")),
        indexingToken_{esConsumes<>(iConfig.getParameter<edm::ESInputTag>("indexingSource"))},
        cellMappingToken_{esConsumes<>(iConfig.getParameter<edm::ESInputTag>("cellMappingSource"))},
        g4AccumulatorToken_{produces()},
        nMaxReadoutCells_(0) {
    //add monitor counters if required
    if (produceExtra_)
      accMonitorToken_ = produces("HGCalSimHitAccumulatorCounters");

    //configure Geant4 hit sources to handle
    auto g4HitCollections = iConfig.getParameter<VPSet>("g4HitsConfig");
    hitConsumerConfig_.reserve(g4HitCollections.size());
    for (const auto& cIt : g4HitCollections) {
      auto d = cIt.getParameter<int>("det");
      auto isc = cIt.getParameter<bool>("iscalib");

      //check if it has already been added, just in case
      auto cfgIt = findHitConsumerConfig(d, isc);
      if (cfgIt != hitConsumerConfig_.end()) {
        edm::LogWarning("HGCalSimHitAccumulator")
            << " A configuration for det=" << d << " calib=" << isc << " already exists, skipping this one";
        continue;
      }

      auto tof_delay = cIt.getParameter<double>("tof_delay");
      auto tag = cIt.getParameter<edm::InputTag>("hits");
      hitConsumerConfig_.push_back(PCaloHitConfig(d, isc, tof_delay, tag, consumes<edm::PCaloHitContainer>(tag)));
    }
  }

  //
  std::vector<PCaloHitConfig>::const_iterator HGCalSimHitAccumulator::findHitConsumerConfig(int32_t det,
                                                                                            bool iscalib) const {
    return std::find_if(
        hitConsumerConfig_.begin(), hitConsumerConfig_.end(), [det, iscalib](PCaloHitConfig const& config) {
          return config.det == det && config.iscalib == iscalib;
        });
  }

  //
  void HGCalSimHitAccumulator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<bool>("produceExtra", false)->setComment("Produce extra counters for monitoring");
    desc.add<edm::ESInputTag>("indexingSource")->setComment("Label for cell dense indexer");
    desc.add<edm::ESInputTag>("cellMappingSource")->setComment("Label for cell information type information");

    edm::ParameterSet hitDefault;
    edm::ParameterSetDescription hitDesc;
    hitDefault.addParameter<int32_t>("det", 8);
    hitDesc.add<int32_t>("det")->setComment(
        "DetId detector codes expected for the sources of PCaloHits (see "
        "https://github.com/cms-sw/cmssw/blob/master/DataFormats/DetId/interface/DetId.h#L24)");
    hitDefault.addParameter<bool>("iscalib", false);
    hitDesc.add<bool>("iscalib")->setComment("is calibration channel?");
    hitDefault.addParameter<double>("tof_delay", -9);
    hitDesc.add<double>("tof_delay")->setComment("TOF delay [ns] to apply to PCaloHit");
    hitDefault.addParameter<edm::InputTag>("hits", edm::InputTag("g4SimHits", "HGCHitsEE"));
    hitDesc.add<edm::InputTag>("hits")->setComment("Label for source of PCaloHits (Geant4)");
    desc.addVPSet("g4HitsConfig", hitDesc, {hitDefault})->setComment("Configuration of PCaloHits to consume");

    descriptions.addWithDefaultLabel(desc);
  }

  //
  int32_t HGCalSimHitAccumulator::buildGeoToDenseIndexMap(const hgcal::HGCalDenseIndexInfoDevice& denseIndexInfo,
                                                          const hgcal::HGCalMappingCellParamHost& cellInfo) const {
    //loop over the indices and map the DetId for the physical cells
    auto const& cellInfo_view = cellInfo.const_view();
    auto const& denseIndexInfo_view = denseIndexInfo.const_view();
    int32_t ndenseIndices = denseIndexInfo_view.metadata().size();

    for (int32_t i = 0; i < ndenseIndices; i++) {
      auto row = denseIndexInfo_view[i];

      //cell type: if it's an unconnected channel just skip it
      auto cell_info_idx = row.cellInfoIdx();
      auto chType = cellInfo_view[cell_info_idx].t();
      if (chType < 0)
        continue;
      bool iscalib = (chType == 0);

      //check if det id is known before mapping it
      const auto k = row.detid();
      const auto det = DetId(k).det();

      //try adding to the dense map
      auto cfgIt = findHitConsumerConfig(det, iscalib);
      if (cfgIt == hitConsumerConfig_.end())
        continue;

      cfgIt->det2dense[k] = i;
    }

    return ndenseIndices;
  }

  //
  void HGCalSimHitAccumulator::produce(edm::StreamID sid,
                                       device::Event& iEvent,
                                       device::EventSetup const& iSetup) const {
    //this is executed only once
    std::call_once(mapInitFlag_, [&] {
      auto const& denseIndexInfo = iSetup.getData(indexingToken_);
      auto const& cellInfo = iSetup.getData(cellMappingToken_);
      nMaxReadoutCells_ = buildGeoToDenseIndexMap(denseIndexInfo, cellInfo);
    });

    //declare collection to produce and initialize with zeros
    hgcaldigi::HGCalSimHitAccumHost g4Accumulators(iEvent.queue(), nMaxReadoutCells_);
    auto accumView = g4Accumulators.view();
    for (int32_t i = 0; i < nMaxReadoutCells_; i++) {
      auto row = accumView[i];
      row.sumE().setZero();
      row.sumExT().setZero();
    }

    //wrap accumulation procedure
    const auto accumulateHit = [&accumView, this](auto const& hit, const auto& cfg, auto& m) {
      const uint32_t did = static_cast<uint32_t>(hit.id());

      const float energy = hit.energy();
      m.en += energy;

      //check detector is of interest
      const auto idxIt = cfg.det2dense.find(did);
      HGCSiliconDetId siid(did);
      if (idxIt == cfg.det2dense.end()) {
        ++m.nHitsUnknown;
        m.enUnknown += energy;
        return;
      }

      //add to accumulator if within time window
      const float tof = hit.time() + cfg.tof_delay;
      const int32_t bx = static_cast<int32_t>(std::roundl(tof / nsPerBx_));
      const int32_t tidx = bx + hgcaldigi::kInTimeBin;
      if (tidx < 0 || tidx >= hgcaldigi::kTimeBins) {
        ++m.nHitsRejected;
        m.enRejected += energy;
        return;
      }

      auto row = accumView[idxIt->second];
      row.sumE()(tidx, 0) += energy;
      row.sumExT()(tidx, 0) += energy * tof;

      if (tidx == hgcaldigi::kInTimeBin) {
        ++m.nHitsIT;
        m.enIT += energy;
      } else {
        ++m.nHitsOOT;
        m.enOOT += energy;
      }
    };  //end accumulation wrapper for a single hit

    //loop over geant4 hit collections and accumulate (monitor counts and put also in the event if needed)
    hgcaldigi::HGCalSimHitAccumulatorMonitorCollection accMonitor;
    accMonitor.reserve(hitConsumerConfig_.size());
    for (auto const& cfg : hitConsumerConfig_) {
      const auto det = cfg.det;
      const auto isc = cfg.iscalib;
      const auto token = cfg.tkn;

      hgcaldigi::HGCalSimHitAccumulatorMonitor m(det, isc);

      try {
        const auto& hit_collection = iEvent.get(token);

        m.ncellsMapped = cfg.det2dense.size();
        m.nHits = hit_collection.size();

        std::for_each(hit_collection.begin(), hit_collection.end(), [&](auto const& h) { accumulateHit(h, cfg, m); });

      } catch (cms::Exception& e) {
        edm::LogWarning("HGCalSimHitAccumulator") << e.what();
      }

      accMonitor.push_back(m);

    }  //end geantl4 inputs loop

    iEvent.emplace(g4AccumulatorToken_, std::move(g4Accumulators));
    if (produceExtra_) {
      iEvent.emplace(accMonitorToken_, accMonitor);
    }
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "DataFormats/HGCalDigi/interface/alpaka/HGCalSimHitAccumDevice.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalSimHitAccumulator);
