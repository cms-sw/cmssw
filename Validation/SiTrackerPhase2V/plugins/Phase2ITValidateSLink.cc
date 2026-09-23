// -*- C++ -*-
// Package:    Validation/SiTrackerPhase2V
// Class:      Phase2ITValidateSLink
// Description: Validate the per-slink occupancy DQM
//
// Author: Lacey Dishman, Sihyun Jeon (Boston University)
// Written: August 2026

#include <algorithm>
#include <string>
#include <vector>

#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Transition.h"

class Phase2ITValidateSLink : public DQMEDAnalyzer {
public:
  explicit Phase2ITValidateSLink(const edm::ParameterSet& iConfig);
  ~Phase2ITValidateSLink() override = default;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookDTCHistos(DQMStore::IBooker& ibooker);

  const edm::EDGetTokenT<RawDataBuffer> rawDataToken_;
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;

  const double scaleTBPX_;
  const double scaleTFPX_;
  const double scaleTEPX_;
  const double trigger_rate_;
  const double slink_bandwidth_;
  const double dataSizeMax_;  // upper edge of the per-DTC data size axis [kb]
  const std::string folder_;

  const TrackerDetToDTCELinkCablingMap* cablingMap_ = nullptr;

  // dtcIds_ / nDTCs_ are populated from the cabling map in dqmBeginRun
  int nDTCs_ = 36;
  int nslinksPerDTC_ = 16;
  std::vector<int> dtcIds_;

  MonitorElement* me_nEvents_ = nullptr;  // event counter, used by the harvester for normalization
  MonitorElement* me_slinkOccupancy_ = nullptr;

  // Per-DTC: 16 SLink bins, <occupancy> with across-event RMS error bars
  std::vector<MonitorElement*> mes_slinkOccupancyPerDTC_;
  std::vector<MonitorElement*> mes_slinkSpectrumOccupancyPerDTC_;

  // Per-DTC data size per event, summed over the 16 SLinks of the DTC
  std::vector<MonitorElement*> mes_dataSizePerDTC_;

  // Cross-DTC overview + 2D heatmap
  MonitorElement* me_slinkOccupancyByDTC_ = nullptr;
  MonitorElement* me_slinkOccupancyMap_ = nullptr;

  // Full spectrum occupancy per DTC
  MonitorElement* me_slinkOccupancyVsDTC_ = nullptr;
};

Phase2ITValidateSLink::Phase2ITValidateSLink(const edm::ParameterSet& iConfig)
    : rawDataToken_(consumes<RawDataBuffer>(iConfig.getParameter<edm::InputTag>("src"))),
      cablingMapToken_(
          esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd, edm::Transition::BeginRun>()),
      scaleTBPX_(iConfig.getUntrackedParameter<double>("scaleTBPX", 1)),
      scaleTFPX_(iConfig.getUntrackedParameter<double>("scaleTFPX", 1)),
      scaleTEPX_(iConfig.getUntrackedParameter<double>("scaleTEPX", 1)),
      trigger_rate_(iConfig.getUntrackedParameter<double>("trigger_rate", 750.0e3)),
      slink_bandwidth_(iConfig.getUntrackedParameter<double>("slink_bandwidth", 25.0e9)),
      dataSizeMax_(iConfig.getUntrackedParameter<double>("dataSizeMax", 800.)),
      folder_(iConfig.getUntrackedParameter<std::string>("folder", "InnerTrackerV/SLink")) {
  edm::LogInfo("Phase2ITValidateSLink") << ">>> Construct Phase2ITValidateSLink";
}

void Phase2ITValidateSLink::dqmBeginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  cablingMap_ = &iSetup.getData(cablingMapToken_);

  auto known = cablingMap_->getKnownDTCIdsWithIndex();
  dtcIds_.assign(known.size(), 0);
  for (const auto& [idx, dtcId] : known)
    dtcIds_[idx] = dtcId;
  nDTCs_ = dtcIds_.size();
}

void Phase2ITValidateSLink::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(folder_);

  me_nEvents_ = ibooker.book1D("nEvents", "Processed events;;Events", 1, 0., 1.);

  me_slinkOccupancy_ =
      ibooker.book1D("slinkOccupancy", "Full Spectrum SLink Occupancy;Occupancy;Per-event SLink entries", 80, 0., 1.6);

  bookDTCHistos(ibooker);

  // Cross-DTC overview: 1 bin per DTC, <occupancy> with across-event RMS error bars
  me_slinkOccupancyByDTC_ = ibooker.bookProfile("slinkOccupancyByDTC",
                                                "Mean SLink Occupancy Averaged Over DTCs;DTC;<Occupancy>",
                                                nDTCs_,
                                                -0.5,
                                                nDTCs_ - 0.5,
                                                0.,
                                                1.6);
  me_slinkOccupancyByDTC_->getTH1()->SetMinimum(0);
  me_slinkOccupancyByDTC_->getTH1()->SetMaximum(1.6);

  me_slinkOccupancyMap_ = ibooker.bookProfile2D("slinkOccupancyMap",
                                                "Mean SLink Occupancy;DTC;SLink Index;<Occupancy>",
                                                nDTCs_,
                                                -0.5,
                                                nDTCs_ - 0.5,
                                                nslinksPerDTC_,
                                                -0.5,
                                                nslinksPerDTC_ - 0.5,
                                                0.,
                                                1.6);
  me_slinkOccupancyMap_->getTH1()->SetStats(0);
  me_slinkOccupancyMap_->getTH1()->SetMinimum(0);
  me_slinkOccupancyMap_->getTH1()->SetMaximum(1.6);
  me_slinkOccupancyMap_->getTH1()->SetOption("COLZ");

  me_slinkOccupancyVsDTC_ = ibooker.book2D(
      "slinkOccupancyVsDTC", "Full Spectrum SLink Occupancy;DTC;Occupancy", nDTCs_, -0.5, nDTCs_ - 0.5, 80, 0., 1.6);
  me_slinkOccupancyVsDTC_->getTH1()->SetStats(0);
  me_slinkOccupancyVsDTC_->getTH1()->SetOption("COLZ");

  // Label DTC axes with the real DTC numbers (11-19, 21-29, ...) instead of index
  for (int i = 0; i < nDTCs_; ++i) {
    me_slinkOccupancyByDTC_->setBinLabel(i + 1, std::to_string(dtcIds_[i]), 1);
    me_slinkOccupancyMap_->setBinLabel(i + 1, std::to_string(dtcIds_[i]), 1);
    me_slinkOccupancyVsDTC_->setBinLabel(i + 1, std::to_string(dtcIds_[i]), 1);
  }
}

void Phase2ITValidateSLink::bookDTCHistos(DQMStore::IBooker& ibooker) {
  mes_slinkOccupancyPerDTC_.assign(nDTCs_, nullptr);
  mes_slinkSpectrumOccupancyPerDTC_.assign(nDTCs_, nullptr);
  mes_dataSizePerDTC_.assign(nDTCs_, nullptr);

  for (int i = 0; i < nDTCs_; ++i) {
    mes_slinkOccupancyPerDTC_[i] = ibooker.bookProfile(
        ("slinkOccupancyPerDTC_" + std::to_string(dtcIds_[i])).c_str(),
        ("Mean SLink Occupancy, DTC " + std::to_string(dtcIds_[i]) + ";SLink Index;<Occupancy>").c_str(),
        nslinksPerDTC_,
        -0.5,
        nslinksPerDTC_ - 0.5,
        0.,
        1.6);
    mes_slinkOccupancyPerDTC_[i]->getTH1()->SetMinimum(0);
    mes_slinkOccupancyPerDTC_[i]->getTH1()->SetMaximum(1.6);

    mes_slinkSpectrumOccupancyPerDTC_[i] = ibooker.book1D(
        ("slinkSpectrumOccupancyPerDTC_" + std::to_string(dtcIds_[i])).c_str(),
        ("Full Spectrum SLink Occupancy, DTC " + std::to_string(dtcIds_[i]) + ";Occupancy;Per-event SLink entries")
            .c_str(),
        80,
        0.,
        1.6);

    mes_dataSizePerDTC_[i] =
        ibooker.book1D(("dataSizePerDTC_" + std::to_string(dtcIds_[i])).c_str(),
                       ("Data Size per Event, DTC " + std::to_string(dtcIds_[i]) + ";data size [kb];Events").c_str(),
                       100,
                       0.,
                       dataSizeMax_);
  }
}

void Phase2ITValidateSLink::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<RawDataBuffer> raw;
  iEvent.getByToken(rawDataToken_, raw);
  if (!raw.isValid())
    return;

  me_nEvents_->Fill(0.5);

  // Data size of this event per DTC, accumulated over its SLinks
  std::vector<double> dtcBits(nDTCs_, 0.);

  // fedId = dtcIndex * SLINKS_PER_DTC + slinkId.
  for (auto it = raw->map().cbegin(); it != raw->map().cend(); ++it) {
    const int fid = static_cast<int>(it->first);

    const int dtcIdx = fid / nslinksPerDTC_;
    const int slinkId = fid % nslinksPerDTC_;
    if (dtcIdx >= nDTCs_)
      continue;

    const double fragBits = static_cast<double>(raw->fragmentData(it).size()) * 8.0;
    const double occupancy = (fragBits * trigger_rate_) / slink_bandwidth_;

    me_slinkOccupancy_->Fill(occupancy);
    mes_slinkOccupancyPerDTC_[dtcIdx]->Fill(slinkId, occupancy);
    mes_slinkSpectrumOccupancyPerDTC_[dtcIdx]->Fill(occupancy);
    me_slinkOccupancyByDTC_->Fill(dtcIdx, occupancy);
    me_slinkOccupancyMap_->Fill(dtcIdx, slinkId, occupancy);
    me_slinkOccupancyVsDTC_->Fill(dtcIdx, occupancy);
    dtcBits[dtcIdx] += fragBits;
  }

  for (int i = 0; i < nDTCs_; ++i) {
    if (dtcBits[i] > 0.)
      mes_dataSizePerDTC_[i]->Fill(dtcBits[i] / 1000.);  // bits -> kb
  }
}

void Phase2ITValidateSLink::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("BitStreamToRawProducer"));
  desc.addUntracked<double>("scaleTBPX", 1);
  desc.addUntracked<double>("scaleTFPX", 1);
  desc.addUntracked<double>("scaleTEPX", 1);
  desc.addUntracked<double>("trigger_rate", 750.0e3);
  desc.addUntracked<double>("slink_bandwidth", 25.0e9);
  desc.addUntracked<double>("dataSizeMax", 800.);
  desc.addUntracked<std::string>("folder", "InnerTrackerV/SLink");
  descriptions.add("Phase2ITValidateSLink", desc);
}

DEFINE_FWK_MODULE(Phase2ITValidateSLink);
