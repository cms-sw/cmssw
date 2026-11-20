#include <iostream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cmath>
#include <algorithm>

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

// FIXME: CALIBRATION CHANNELS!!!
//Branch 27 of Events tree: PCaloHits_g4SimHits_CalibrationHGCHitsEE_SIM. Total size = 4222 Compressed size = 592
//Branch 28 of Events tree: PCaloHits_g4SimHits_CalibrationHGCHitsHEback_SIM. Total size = 4318 Compressed size = 608
//Branch 29 of Events tree: PCaloHits_g4SimHits_CalibrationHGCHitsHEfront_SIM. Total size = 4342 Compressed size = 612

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
    @short reads the electronics dense indexing and builds an internal DetId -> DenseIndex mapping
    The mapping is only built once in a run. The `ncellsMapped_` variable is used to track if the map is available or not
    */
    int32_t buildGeoToDenseIndexMap(const hgcal::HGCalDenseIndexInfoDevice&,
                                    const hgcal::HGCalMappingCellParamHost&) const;

    const device::ESGetToken<hgcal::HGCalDenseIndexInfoDevice, HGCalDenseIndexInfoRcd> indexingToken_;
    const edm::ESGetToken<hgcal::HGCalMappingCellParamHost, HGCalElectronicsMappingRcd> cellMappingToken_;
    std::unordered_map<int32_t, edm::EDGetTokenT<edm::PCaloHitContainer>> g4HitsTokens_;
    std::unordered_map<int32_t, float> g4TOFdelay_;
    const edm::EDPutTokenT<hgcaldigi::HGCalSimHitAccumHost> g4AccumulatorToken_;
    mutable std::once_flag mapInitFlag_;
    mutable std::unordered_map<int32_t, std::unordered_map<uint32_t, int32_t>> det2dense_;
    mutable int32_t nMaxReadoutCells_ = 0;
    static constexpr float nsPerBx_ = 25.0f;
  };

  //
  HGCalSimHitAccumulator::HGCalSimHitAccumulator(const edm::ParameterSet& iConfig)
      : EDProducer<>(iConfig),
        indexingToken_{esConsumes<>(iConfig.getParameter<edm::ESInputTag>("indexingSource"))},
        cellMappingToken_{esConsumes<>(iConfig.getParameter<edm::ESInputTag>("cellMappingSource"))},
        g4HitsTokens_{},
        g4AccumulatorToken_{produces()},
        nMaxReadoutCells_(0) {
    //configure Geant4 hit sources to handle
    auto g4HitSubDet = iConfig.getParameter<std::vector<int32_t>>("g4HitsDets");
    auto g4HitInputs = iConfig.getParameter<std::vector<edm::InputTag>>("g4Hits");
    auto g4TOFDelays = iConfig.getParameter<std::vector<double>>("g4TOFDelays");
    if (g4HitSubDet.size() != g4HitInputs.size() || g4TOFDelays.size() != g4HitInputs.size())
      throw cms::Exception(
          "HGCalSimHitAccumulator: Geant4 hit sub-detector codes, sources and time-of-flight delays are not fully "
          "consistent in size!");
    for (size_t i = 0; i < g4HitSubDet.size(); i++) {
      auto det = g4HitSubDet[i];
      g4HitsTokens_[det] = consumes<edm::PCaloHitContainer>(g4HitInputs[i]);
      g4TOFdelay_[det] = static_cast<float>(g4TOFDelays[i]);
      edm::LogWarning("") << "will consume " << g4HitInputs[i] << " PCaloHits collection for det=" << det
                          << " TOF delay=" << g4TOFDelays[i];
    }
  }

  //
  void HGCalSimHitAccumulator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<int32_t>>("g4HitsDets")
        ->setComment(
            "DetId detector codes expected for the sources of PCaloHits (see "
            "https://github.com/cms-sw/cmssw/blob/master/DataFormats/DetId/interface/DetId.h#L24)");
    desc.add<std::vector<edm::InputTag>>("g4Hits")->setComment("Label for source of PCaloHits (Geant4)");
    desc.add<std::vector<double>>("g4TOFDelays")->setComment("TOF delays to apply to PCaloHits (Geant4)");
    desc.add<edm::ESInputTag>("indexingSource")->setComment("Label for cell dense indexer");
    desc.add<edm::ESInputTag>("cellMappingSource")->setComment("Label for cell information type information");
    descriptions.addWithDefaultLabel(desc);
  }

  //
  int32_t HGCalSimHitAccumulator::buildGeoToDenseIndexMap(const hgcal::HGCalDenseIndexInfoDevice& denseIndexInfo,
                                                          const hgcal::HGCalMappingCellParamHost& cellInfo) const {
    //loop over the indices and map the DetId for the physical cells
    auto const& cellInfo_view = cellInfo.const_view();
    auto const& denseIndexInfo_view = denseIndexInfo.const_view();
    int32_t ndenseIndices = denseIndexInfo_view.metadata().size();
    float nunknown(0.);
    for (int32_t i = 0; i < ndenseIndices; i++) {
      auto row = denseIndexInfo.view()[i];

      //cell type: if it's an unconnected channel just skip it
      auto cell_info_idx = row.cellInfoIdx();
      auto chType = cellInfo_view[cell_info_idx].t();
      if (chType < 0)
        continue;

      //check if det id is known before mapping it
      const auto k = row.detid();
      const auto det = DetId(k).det();
      if (g4HitsTokens_.find(det) == g4HitsTokens_.end()) {
        nunknown += 1;
        continue;
      }

      //accept to map this detid
      det2dense_[det][k] = i;
    }

    //sum up final mapped cells (and print some debug if needed)
    int nmapped(0);
    for (const auto& it : det2dense_) {
      int32_t ncells = static_cast<int32_t>(it.second.size());
      edm::LogWarning("") << "Detector " << it.first << " has " << ncells << " cells mapped";
      nmapped += ncells;
    }

    if (ndenseIndices > 0) {
      edm::LogWarning("") << "Fraction of cells mapped:" << 100 * ((float)(nmapped) / ndenseIndices) << "%";
      if (nunknown > 0)
        edm::LogWarning("") << "Fraction of unknown cells " << 100 * ((float)(nunknown) / ndenseIndices) << "%";
    }

    return ndenseIndices;
  }

  //
  void HGCalSimHitAccumulator::produce(edm::StreamID sid,
                                       device::Event& iEvent,
                                       device::EventSetup const& iSetup) const {
    //this is executed only once even if there are concurrent threads
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
    int totalHits(0), totalUnknown(0), totalOOT(0);
    const auto accumulateHit = [&totalUnknown, &totalOOT, &accumView, this](const auto& hit) {
      const uint32_t did = static_cast<uint32_t>(hit.id());
      const auto det = DetId(did).det();

      //check detector is of interest
      const auto detIt = this->det2dense_.find(det);
      if (detIt == this->det2dense_.end()) {
        totalUnknown++;
        return;
      }

      //check if it is known
      const auto idx = detIt->second.find(did);
      if (idx == detIt->second.end()) {
        totalUnknown++;
        return;
      }

      //add to accumulator if within time window
      const float energy = hit.energy();
      const float tof = hit.time() + g4TOFdelay_.at(det);
      const int32_t bx = static_cast<int32_t>(std::roundl(tof / nsPerBx_));
      const int32_t tidx = bx + hgcaldigi::kInTimeBin;
      if (tidx < 0 || tidx >= hgcaldigi::kTimeBins) {
        totalOOT++;
        return;
      }

      auto row = accumView[idx->second];
      row.sumE()(tidx, 0) += energy;
      row.sumExT()(tidx, 0) += energy * tof;
    };  //end accumulation wrapper for a single hit

    //loop over geant4 hit collections and accumulate
    for (auto const& g4It : g4HitsTokens_) {
      const auto k = g4It.first;
      const auto token = g4It.second;
      const auto& hit_collection = iEvent.get(token);
      totalHits += hit_collection.size();
      edm::LogWarning("") << "Det=" << k << " has " << hit_collection.size() << " hits";
      std::for_each(hit_collection.begin(), hit_collection.end(), accumulateHit);
    }  //end g4 inputs loop

    if (totalHits > 0 && totalUnknown + totalOOT > 0)
      edm::LogWarning("") << "Hit rejection fractions: " << 100 * ((float)(totalUnknown) / totalHits) << "% unmapped | "
                          << 100 * ((float)(totalOOT) / totalHits) << "% out-of-time window";

    iEvent.emplace(g4AccumulatorToken_, std::move(g4Accumulators));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "DataFormats/HGCalDigi/interface/alpaka/HGCalSimHitAccumDevice.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(HGCalSimHitAccumulator);
