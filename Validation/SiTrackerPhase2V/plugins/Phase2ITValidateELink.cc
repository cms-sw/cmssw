// -*- C++ -*-
// Package:    Validation/SiTrackerPhase2V
// Class:      Phase2ITValidateELink
// Description: Validate the per-elink occupancy DQM
//
// Author: Lacey Dishman, Sihyun Jeon (Boston University)
// Written: August 2026

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITAuroraBitStream.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Transition.h"

class Phase2ITValidateELink : public DQMEDAnalyzer {
public:
  explicit Phase2ITValidateELink(const edm::ParameterSet& iConfig);
  ~Phase2ITValidateELink() override = default;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookSectionHistos(DQMStore::IBooker& ibooker);
  void bookSubTypeHistos(DQMStore::IBooker& ibooker);
  static int sectionIndexOf(int section, int layer, int ring);

  const edm::EDGetTokenT<edm::DetSetVector<Phase2ITAuroraBitStream>> auroraToken_;
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;

  const double scaleTBPX_;
  const double scaleTFPX_;
  const double scaleTEPX_;
  const double trigger_rate_;
  const double elink_bandwidth_;
  const std::string folder_;

  const TrackerDetToDTCELinkCablingMap* cablingMap_ = nullptr;

  MonitorElement* me_elinkOccupancy_ = nullptr;

  // Cross-region overviews + 2D maps
  std::vector<std::string> sectionLabels_;
  MonitorElement* me_elinkOccupancyBySection_ = nullptr;
  MonitorElement* me_elinkOccupancyBySubType_ = nullptr;
  MonitorElement* me_elinkOccupancyVsSection_ = nullptr;
  MonitorElement* me_elinkOccupancyVsSubType_ = nullptr;
  // One entry per NE stream group (one Aurora frame), NOT PER EVENT
  MonitorElement* me_nStreamGroups_ = nullptr;

  static constexpr int nSections_ = 13;  // 4 TBPX + 4 TFPX + 5 TEPX
  std::vector<MonitorElement*> mes_elinkOccupancyPerSection_;

  std::map<std::pair<int, int>, unsigned int> sectionToIndex_;  // {sectionCode, layer/ring} -> idx

  std::vector<int> subTypeVals_;
  std::vector<MonitorElement*> mes_elinkOccupancyPerSubType_;
  std::map<int, unsigned int> subTypeToIndex_;

  std::map<uint32_t, int> moduleToIndex_;  // detId -> module index (0..3999)
  int nModules_ = 0;
  static constexpr int maxELinksPerModule_ = 6;
  MonitorElement* me_elinkOccupancyMap_ = nullptr;

  // Per-section quadrant maps: Y = quadrant, X = elinks of each module in the quadrant
  struct QuadSlot {
    int sec = -1;   // section index
    int row = -1;   // quadrant row
    int xbase = 0;  // first X slot of one module's elinks
    int width = 0;  // nElinks for one module
  };
  std::map<uint32_t, QuadSlot> moduleSlot_;  // detId -> x-axis slot, further split into N elinks
  std::vector<int> quadrantVals_;            // sorted by DTC decade: 11..19, 21..29, 31..39, 41..49
  std::vector<int> sectionMapNX_;            // per section, X bins = max over quadrants of summed module ELinks
  std::vector<MonitorElement*> mes_elinkOccupancyMapPerSection_;

  // eLinkOccupancyMap split by quadrant, module index running within the quadrant
  std::map<uint32_t, std::pair<int, int>> moduleQuad_;  // detId -> (quadrant row, index in quadrant)
  std::vector<int> nModulesPerQuad_;
  std::vector<MonitorElement*> mes_elinkOccupancyMapQuad_;
};

Phase2ITValidateELink::Phase2ITValidateELink(const edm::ParameterSet& iConfig)
    : auroraToken_(
          consumes<edm::DetSetVector<Phase2ITAuroraBitStream>>(iConfig.getParameter<edm::InputTag>("auroraBitStream"))),
      cablingMapToken_(
          esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd, edm::Transition::BeginRun>()),
      scaleTBPX_(iConfig.getUntrackedParameter<double>("scaleTBPX", 1)),
      scaleTFPX_(iConfig.getUntrackedParameter<double>("scaleTFPX", 1)),
      scaleTEPX_(iConfig.getUntrackedParameter<double>("scaleTEPX", 1)),
      trigger_rate_(iConfig.getUntrackedParameter<double>("trigger_rate", 750.0e3)),
      elink_bandwidth_(iConfig.getUntrackedParameter<double>("elink_bandwidth", 1.28e9)),
      folder_(iConfig.getUntrackedParameter<std::string>("folder", "InnerTrackerV/ELink")) {
  edm::LogInfo("Phase2ITValidateELink") << ">>> Construct Phase2ITValidateELink";
}

int Phase2ITValidateELink::sectionIndexOf(int section, int layer, int ring) {
  using Section = TrackerDetToDTCELinkCablingMap::Section;
  if (section == static_cast<int>(Section::TBPX) && layer >= 1 && layer <= 4)
    return layer - 1;
  if (section == static_cast<int>(Section::TFPX) && ring >= 1 && ring <= 4)
    return 4 + (ring - 1);
  if (section == static_cast<int>(Section::TEPX) && ring >= 1 && ring <= 5)
    return 8 + (ring - 1);
  return -1;
}

void Phase2ITValidateELink::dqmBeginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  cablingMap_ = &iSetup.getData(cablingMapToken_);

  std::set<int> subtypes;
  std::set<uint32_t> modules;
  std::set<int> quadrants;
  std::map<int, std::map<int, std::vector<uint32_t>>> secQuadModules;
  for (uint32_t detId : cablingMap_->getKnownDetIds()) {
    if (!cablingMap_->hasModuleInfo(detId))
      continue;
    const auto& info = cablingMap_->getModuleInfo(detId);
    subtypes.insert(static_cast<int>(info.subtype));
    modules.insert(detId);

    const auto elinks = cablingMap_->detIdToDTCELinkId(detId);
    if (elinks.first == elinks.second)
      continue;
    const int dtcId = static_cast<int>(elinks.first->second.dtc_id());
    // DTC 11..19 -> 1, 21..29 -> 2, 31..39 -> 3, 41..49 -> 4
    quadrants.insert(dtcId / 10);
    const int secIdx =
        sectionIndexOf(static_cast<int>(info.section), static_cast<int>(info.layer), static_cast<int>(info.ring));
    if (secIdx < 0)
      continue;
    secQuadModules[secIdx][dtcId / 10].push_back(detId);
  }
  subTypeVals_.assign(subtypes.begin(), subtypes.end());

  quadrantVals_.assign(quadrants.begin(), quadrants.end());
  int m = 0;
  moduleQuad_.clear();
  nModulesPerQuad_.assign(quadrantVals_.size(), 0);
  for (uint32_t detId : modules) {
    moduleToIndex_[detId] = m++;
    const auto elinks = cablingMap_->detIdToDTCELinkId(detId);
    if (elinks.first == elinks.second)
      continue;
    const auto q =
        std::find(quadrantVals_.begin(), quadrantVals_.end(), static_cast<int>(elinks.first->second.dtc_id()) / 10);
    if (q == quadrantVals_.end())
      continue;
    const int row = static_cast<int>(q - quadrantVals_.begin());
    moduleQuad_[detId] = {row, nModulesPerQuad_[row]++};
  }
  nModules_ = m;
  moduleSlot_.clear();
  sectionMapNX_.assign(nSections_, 0);
  for (auto& [secIdx, byQuad] : secQuadModules) {
    for (auto& [decade, ids] : byQuad) {
      std::sort(ids.begin(), ids.end());
      const int row =
          static_cast<int>(std::find(quadrantVals_.begin(), quadrantVals_.end(), decade) - quadrantVals_.begin());
      int xbase = 0;
      for (uint32_t id : ids) {
        const int width = std::max<int>(1, cablingMap_->getModuleInfo(id).nElinks);
        moduleSlot_[id] = QuadSlot{secIdx, row, xbase, width};
        xbase += width;
      }
      sectionMapNX_[secIdx] = std::max(sectionMapNX_[secIdx], xbase);
    }
  }
}

void Phase2ITValidateELink::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(folder_);

  me_nStreamGroups_ = ibooker.book1D("nStreamGroups", "Processed NE stream groups;;Stream groups", 1, 0., 1.);

  me_elinkOccupancy_ = ibooker.book1D(
      "eLinkOccupancy", "Full Spectrum ELink Occupancy;Occupancy;ELink entries per stream group", 80, 0., 1.6);

  bookSectionHistos(ibooker);
  bookSubTypeHistos(ibooker);

  const int nSub = static_cast<int>(subTypeVals_.size());

  me_elinkOccupancyMap_ = ibooker.bookProfile2D("eLinkOccupancyMap",
                                                "Mean ELink Occupancy;Module Index;ELink Index;<Occupancy>",
                                                nModules_,
                                                -0.5,
                                                nModules_ - 0.5,
                                                maxELinksPerModule_,
                                                -0.5,
                                                maxELinksPerModule_ - 0.5,
                                                0.,
                                                0.);
  me_elinkOccupancyMap_->getTH1()->SetStats(0);

  mes_elinkOccupancyMapQuad_.assign(quadrantVals_.size(), nullptr);
  for (size_t r = 0; r < quadrantVals_.size(); ++r) {
    const std::string q = "Q" + std::to_string(quadrantVals_[r]);
    const int nx = std::max(1, nModulesPerQuad_[r]);
    MonitorElement* me = ibooker.bookProfile2D(
        ("eLinkOccupancyMap_" + q).c_str(),
        ("Mean ELink Occupancy, " + q + ";Module Index in " + q + ";ELink Index;<Occupancy>").c_str(),
        nx,
        -0.5,
        nx - 0.5,
        maxELinksPerModule_,
        -0.5,
        maxELinksPerModule_ - 0.5,
        0.,
        0.);
    me->getTH1()->SetStats(0);
    mes_elinkOccupancyMapQuad_[r] = me;
  }

  // 1 bin per section, <occupancy> with across-event RMS error bars
  me_elinkOccupancyBySection_ = ibooker.bookProfile("eLinkOccupancyBySection",
                                                    "Mean ELink Occupancy Averaged Over Sections;Section;<Occupancy>",
                                                    nSections_,
                                                    -0.5,
                                                    nSections_ - 0.5,
                                                    0.,
                                                    0.);
  me_elinkOccupancyBySection_->getTH1()->SetMinimum(0);
  me_elinkOccupancyBySection_->getTH1()->SetMaximum(1.6);

  // 1 bin per subtype, <occupancy> with across-event RMS error bars
  me_elinkOccupancyBySubType_ = ibooker.bookProfile("eLinkOccupancyBySubType",
                                                    "Mean ELink Occupancy Averaged Over SubTypes;SubType;<Occupancy>",
                                                    nSub,
                                                    -0.5,
                                                    nSub - 0.5,
                                                    0.,
                                                    0.);
  me_elinkOccupancyBySubType_->getTH1()->SetMinimum(0);
  me_elinkOccupancyBySubType_->getTH1()->SetMaximum(1.6);

  me_elinkOccupancyVsSection_ = ibooker.book2D("eLinkOccupancyVsSection",
                                               "Full Spectrum ELink Occupancy;Section;Occupancy",
                                               nSections_,
                                               -0.5,
                                               nSections_ - 0.5,
                                               80,
                                               0.,
                                               1.6);
  me_elinkOccupancyVsSection_->getTH1()->SetStats(0);
  me_elinkOccupancyVsSection_->getTH1()->SetOption("COLZ");

  me_elinkOccupancyVsSubType_ = ibooker.book2D("eLinkOccupancyVsSubType",
                                               "Full Spectrum ELink Occupancy;SubType;Occupancy",
                                               nSub,
                                               -0.5,
                                               nSub - 0.5,
                                               80,
                                               0.,
                                               1.6);
  me_elinkOccupancyVsSubType_->getTH1()->SetStats(0);
  me_elinkOccupancyVsSubType_->getTH1()->SetOption("COLZ");

  // Label section axes (TBPX_L1.., TFPX_R1.., TEPX_R1..) and subtype axes (subtype IDs)
  for (int i = 0; i < nSections_; ++i) {
    me_elinkOccupancyBySection_->setBinLabel(i + 1, sectionLabels_[i], 1);
    me_elinkOccupancyVsSection_->setBinLabel(i + 1, sectionLabels_[i], 1);
  }

  for (int i = 0; i < nSub; ++i) {
    me_elinkOccupancyBySubType_->setBinLabel(i + 1, std::to_string(subTypeVals_[i]), 1);
    me_elinkOccupancyVsSubType_->setBinLabel(i + 1, std::to_string(subTypeVals_[i]), 1);
  }

  const int nQuad = std::max<int>(1, static_cast<int>(quadrantVals_.size()));
  mes_elinkOccupancyMapPerSection_.assign(nSections_, nullptr);
  for (int i = 0; i < nSections_; ++i) {
    if (sectionMapNX_[i] <= 0)
      continue;
    MonitorElement* me = ibooker.bookProfile2D(
        ("eLinkOccupancyMapPerSection_" + sectionLabels_[i]).c_str(),
        ("Mean ELink Occupancy, " + sectionLabels_[i] + ";eLink Index;Quadrant;<Occupancy>").c_str(),
        sectionMapNX_[i],
        -0.5,
        sectionMapNX_[i] - 0.5,
        nQuad,
        -0.5,
        nQuad - 0.5,
        0.,
        0.);
    me->getTH1()->SetStats(0);
    me->getTH1()->SetOption("COLZ");
    me->getTH1()->SetMinimum(0);
    me->getTH1()->SetMaximum(1.6);
    for (int r = 0; r < static_cast<int>(quadrantVals_.size()); ++r)
      me->setBinLabel(r + 1, "Q" + std::to_string(quadrantVals_[r]), 2);

    mes_elinkOccupancyMapPerSection_[i] = me;
  }
}

void Phase2ITValidateELink::bookSectionHistos(DQMStore::IBooker& ibooker) {
  using Section = TrackerDetToDTCELinkCablingMap::Section;
  mes_elinkOccupancyPerSection_.assign(nSections_, nullptr);
  sectionLabels_.assign(nSections_, "");
  int idx = 0;
  for (int L = 1; L <= 4; ++L) {
    mes_elinkOccupancyPerSection_[idx] = ibooker.book1D(
        ("eLinkOccupancyPerSection_TBPX_L" + std::to_string(L)).c_str(),
        ("Full Spectrum ELink Occupancy, TBPX_L" + std::to_string(L) + ";Occupancy;Per-event ELink entries").c_str(),
        80,
        0.,
        1.6);
    sectionLabels_[idx] = "TBPX_L" + std::to_string(L);
    sectionToIndex_[{static_cast<int>(Section::TBPX), L}] = idx++;
  }

  for (int R = 1; R <= 4; ++R) {
    mes_elinkOccupancyPerSection_[idx] = ibooker.book1D(
        ("eLinkOccupancyPerSection_TFPX_R" + std::to_string(R)).c_str(),
        ("Full Spectrum ELink Occupancy, TFPX_R" + std::to_string(R) + ";Occupancy;Per-event ELink entries").c_str(),
        80,
        0.,
        1.6);
    sectionLabels_[idx] = "TFPX_R" + std::to_string(R);
    sectionToIndex_[{static_cast<int>(Section::TFPX), R}] = idx++;
  }

  for (int R = 1; R <= 5; ++R) {
    mes_elinkOccupancyPerSection_[idx] = ibooker.book1D(
        ("eLinkOccupancyPerSection_TEPX_R" + std::to_string(R)).c_str(),
        ("Full Spectrum ELink Occupancy, TEPX_R" + std::to_string(R) + ";Occupancy;Per-event ELink entries").c_str(),
        80,
        0.,
        1.6);
    sectionLabels_[idx] = "TEPX_R" + std::to_string(R);
    sectionToIndex_[{static_cast<int>(Section::TEPX), R}] = idx++;
  }
}

void Phase2ITValidateELink::bookSubTypeHistos(DQMStore::IBooker& ibooker) {
  edm::LogInfo("Phase2ITValidateELink") << "subtypes found: " << subTypeVals_.size();
  mes_elinkOccupancyPerSubType_.assign(subTypeVals_.size(), nullptr);
  for (size_t i = 0; i < subTypeVals_.size(); ++i) {
    mes_elinkOccupancyPerSubType_[i] =
        ibooker.book1D(("eLinkOccupancyPerSubType_" + std::to_string(subTypeVals_[i])).c_str(),
                       ("Full Spectrum ELink Occupancy, SubType " + std::to_string(subTypeVals_[i]) +
                        ";Occupancy;Per-event ELink entries")
                           .c_str(),
                       80,
                       0.,
                       1.6);
    subTypeToIndex_[subTypeVals_[i]] = i;
  }
}

void Phase2ITValidateELink::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<edm::DetSetVector<Phase2ITAuroraBitStream>> handle;
  iEvent.getByToken(auroraToken_, handle);
  if (!handle.isValid() || handle->empty())
    return;

  me_nStreamGroups_->Fill(0.5);

  using Section = TrackerDetToDTCELinkCablingMap::Section;

  for (const auto& detset : *handle) {
    const auto& info = cablingMap_->getModuleInfo(detset.id);
    const int section = static_cast<int>(info.section);
    const double sectionScale = (section == static_cast<int>(Section::TBPX))   ? scaleTBPX_
                                : (section == static_cast<int>(Section::TFPX)) ? scaleTFPX_
                                : (section == static_cast<int>(Section::TEPX)) ? scaleTEPX_
                                                                               : 1.0;
    const int layer = static_cast<int>(info.layer);
    const int ring = static_cast<int>(info.ring);
    const int subtype = static_cast<int>(info.subtype);

    // TBPX keys on layer; TFPX/TEPX key on ring
    std::pair<int, int> key =
        (section == static_cast<int>(Section::TBPX)) ? std::make_pair(section, layer) : std::make_pair(section, ring);
    auto secIt = sectionToIndex_.find(key);
    auto subIt = subTypeToIndex_.find(subtype);
    auto modIt = moduleToIndex_.find(detset.id);

    for (const auto& aurora : detset) {  // aurora = one ELink
      const int elinkIdx = aurora.get_elinkId();
      const int ne = aurora.get_eventsPerStream();  // this is per-elink, not per-stream

      double elinkBits = 0.;
      for (const auto& stream : aurora.get_auroraStreams()) {
        const int bits = static_cast<int>(stream.size());
        elinkBits += bits;
      }

      const double elinkBitsPerEvent = (ne > 0) ? elinkBits / ne : elinkBits;
      const double occupancy = ((elinkBitsPerEvent * trigger_rate_) / (elink_bandwidth_)) * sectionScale;
      me_elinkOccupancy_->Fill(occupancy);

      if (modIt != moduleToIndex_.end())
        me_elinkOccupancyMap_->Fill(modIt->second, elinkIdx, occupancy);
      if (auto qIt = moduleQuad_.find(detset.id); qIt != moduleQuad_.end())
        mes_elinkOccupancyMapQuad_[qIt->second.first]->Fill(qIt->second.second, elinkIdx, occupancy);

      auto slotIt = moduleSlot_.find(detset.id);
      if (slotIt != moduleSlot_.end() && elinkIdx >= 0 && elinkIdx < slotIt->second.width &&
          mes_elinkOccupancyMapPerSection_[slotIt->second.sec] != nullptr) {
        const QuadSlot& slot = slotIt->second;
        mes_elinkOccupancyMapPerSection_[slot.sec]->Fill(slot.xbase + elinkIdx, slot.row, occupancy);
      }

      if (secIt != sectionToIndex_.end()) {
        mes_elinkOccupancyPerSection_[secIt->second]->Fill(occupancy);
        me_elinkOccupancyBySection_->Fill(secIt->second, occupancy);
        me_elinkOccupancyVsSection_->Fill(secIt->second, occupancy);
      }

      if (subIt != subTypeToIndex_.end()) {
        mes_elinkOccupancyPerSubType_[subIt->second]->Fill(occupancy);
        me_elinkOccupancyBySubType_->Fill(subIt->second, occupancy);
        me_elinkOccupancyVsSubType_->Fill(subIt->second, occupancy);
      }
    }
  }
}

void Phase2ITValidateELink::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("auroraBitStream", edm::InputTag("BitStreamToAuroraProducer"));
  desc.addUntracked<double>("scaleTBPX", 1);
  desc.addUntracked<double>("scaleTFPX", 1);
  desc.addUntracked<double>("scaleTEPX", 1);
  desc.addUntracked<double>("trigger_rate", 750.0e3);
  desc.addUntracked<double>("elink_bandwidth", 1.28e9);
  desc.addUntracked<std::string>("folder", "InnerTrackerV/ELink");
  descriptions.add("Phase2ITValidateELink", desc);
}

DEFINE_FWK_MODULE(Phase2ITValidateELink);
