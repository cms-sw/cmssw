// -*- C++ -*-
// Package:    Validation/SiTrackerPhase2V
// Class:      ElinkOccupancyHarvester
// Description: Harvest the per-elink occupancy DQM
//
// Author: Lacey Dishman, Sihyun Jeon (Boston University)
// Written: August 2026

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2HarvestingUtil.h"
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2PlotUtil.h"

#include "TColor.h"
#include "TFile.h"
#include "TH1F.h"
#include "TProfile2D.h"

namespace {
  const std::vector<std::pair<std::string, std::string>> kSectionColors = {{"TBPX_L1", "#000080"},
                                                                           {"TBPX_L2", "#0000FF"},
                                                                           {"TBPX_L3", "#6666FF"},
                                                                           {"TBPX_L4", "#87CEEB"},
                                                                           {"TFPX_R1", "#800000"},
                                                                           {"TFPX_R2", "#FF0000"},
                                                                           {"TFPX_R3", "#FF8000"},
                                                                           {"TFPX_R4", "#FFCC00"},
                                                                           {"TEPX_R1", "#003300"},
                                                                           {"TEPX_R2", "#00AA00"},
                                                                           {"TEPX_R3", "#33EE33"},
                                                                           {"TEPX_R4", "#B0B0B0"},
                                                                           {"TEPX_R5", "#DDDDDD"}};
}  // namespace

class ElinkOccupancyHarvester : public DQMEDHarvester {
public:
  explicit ElinkOccupancyHarvester(const edm::ParameterSet&);
  ~ElinkOccupancyHarvester() override;
  void dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  TProfile2D* sectionMap(DQMStore::IGetter& igetter, const std::string& section) const;
  static std::unique_ptr<TH1F> bookSectionHist(
      const std::string& name, const std::string& title, int nBins, double xMin, double xMax, const std::string& color);

  const edm::ParameterSet config_;
  const std::string topFolder_;
  const std::string occupancyMapName_;
  const bool savePlots_;
  const std::string plotDir_;
  const std::vector<std::string> plotFormats_;
  const double zMax_;
  // Harvested DQM file to compare and draw the delta plots
  const std::string referenceFile_;
  const int deltaBins_;
  const double deltaMin_;
  const double deltaMax_;
};

ElinkOccupancyHarvester::ElinkOccupancyHarvester(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      topFolder_(iConfig.getParameter<std::string>("TopFolder")),
      occupancyMapName_(iConfig.getParameter<std::string>("OccupancyMapName")),
      savePlots_(iConfig.getUntrackedParameter<bool>("savePlots", false)),
      plotDir_(iConfig.getUntrackedParameter<std::string>("plotDir", ".")),
      plotFormats_(iConfig.getUntrackedParameter<std::vector<std::string>>("plotFormats", {"png", "pdf"})),
      zMax_(iConfig.getUntrackedParameter<double>("plotZMax", 1.6)),
      referenceFile_(iConfig.getUntrackedParameter<std::string>("referenceFile", "")),
      deltaBins_(iConfig.getUntrackedParameter<int>("deltaBins", 80)),
      deltaMin_(iConfig.getUntrackedParameter<double>("deltaMin", -0.45)),
      deltaMax_(iConfig.getUntrackedParameter<double>("deltaMax", 0.15)) {}

ElinkOccupancyHarvester::~ElinkOccupancyHarvester() {}

TProfile2D* ElinkOccupancyHarvester::sectionMap(DQMStore::IGetter& igetter, const std::string& section) const {
  MonitorElement* me = igetter.get(topFolder_ + "/eLinkOccupancyMapPerSection_" + section);
  return (me == nullptr) ? nullptr : me->getTProfile2D();
}

std::unique_ptr<TH1F> ElinkOccupancyHarvester::bookSectionHist(
    const std::string& name, const std::string& title, int nBins, double xMin, double xMax, const std::string& color) {
  auto h = std::make_unique<TH1F>(name.c_str(), title.c_str(), nBins, xMin, xMax);
  h->SetDirectory(nullptr);
  h->SetLineColor(TColor::GetColor(color.c_str()));
  return h;
}

void ElinkOccupancyHarvester::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  MonitorElement* occMap = igetter.get(topFolder_ + "/" + occupancyMapName_);
  if (occMap == nullptr) {
    edm::LogWarning("ElinkOccupancyHarvester") << occupancyMapName_ << " not found in " << topFolder_;
    return;
  }
  TProfile2D* prof = occMap->getTProfile2D();

  MonitorElement* nGroupME = igetter.get(topFolder_ + "/nStreamGroups");
  const double nGroups = (nGroupME != nullptr) ? nGroupME->getTH1F()->GetBinContent(1) : 0.;
  if (nGroups <= 0) {
    edm::LogWarning("ElinkOccupancyHarvester")
        << "nStreamGroups missing or empty in " << topFolder_ << "; skipping all normalization.";
  }

  // Normalize the raw 1D occupancy to per stream group
  MonitorElement* elinkOcc = igetter.get(topFolder_ + "/eLinkOccupancy");
  if (elinkOcc != nullptr && nGroups > 0) {
    elinkOcc->getTH1F()->Scale(1.0 / nGroups);
    elinkOcc->getTH1F()->SetOption("HIST");
    elinkOcc->setAxisTitle("ELink entries / stream group", 2);
  }

  // Normalization for per-section / per-subtype
  for (auto* me : igetter.getAllContents(topFolder_)) {
    if (me == nullptr)
      continue;
    if (me->getName().rfind("eLinkOccupancyPer", 0) == 0 && nGroups > 0) {
      me->getTH1F()->Scale(1.0 / nGroups);
      me->getTH1F()->SetOption("HIST");
      me->setAxisTitle("ELink entries / stream group", 2);
    }
  }

  MonitorElement* occVsSection = igetter.get(topFolder_ + "/eLinkOccupancyVsSection");
  if (occVsSection != nullptr && nGroups > 0) {
    occVsSection->getTH2F()->Scale(1.0 / nGroups);
    occVsSection->getTH2F()->SetOption("COLZ");
    occVsSection->setAxisTitle("ELink entries / stream group", 3);
  }

  MonitorElement* occVsSubType = igetter.get(topFolder_ + "/eLinkOccupancyVsSubType");
  if (occVsSubType != nullptr && nGroups > 0) {
    occVsSubType->getTH2F()->Scale(1.0 / nGroups);
    occVsSubType->getTH2F()->SetOption("COLZ");
    occVsSubType->setAxisTitle("ELink entries / stream group", 3);
  }

  // Event-averaged 1D occupancy, one entry per ELink
  ibooker.cd();
  ibooker.setCurrentFolder(topFolder_);
  const edm::ParameterSet avgPSet = config_.getParameter<edm::ParameterSet>("occupancyAvg");
  MonitorElement* occAvg = phase2tkharvestutil::book1DFromPSet(avgPSet, ibooker);
  if (occAvg == nullptr)
    return;

  for (int ix = 1; ix <= prof->GetNbinsX(); ix++) {
    for (int iy = 1; iy <= prof->GetNbinsY(); iy++) {
      if (prof->GetBinEntries(prof->GetBin(ix, iy)) > 0)
        occAvg->Fill(prof->GetBinContent(ix, iy));
    }
  }

  if (savePlots_) {
    TrackerPhase2PlotUtil::PlotConfig plotCfg;
    plotCfg.plotDir = plotDir_;
    plotCfg.formats = plotFormats_;
    plotCfg.occMapNamePrefix = "eLinkOccupancyMap";
    plotCfg.zMax = zMax_;
    int nFiles = TrackerPhase2PlotUtil::saveFolderPlots(igetter, topFolder_, plotCfg);

    // The module x ELink map, one quadrant per row
    std::vector<std::pair<std::string, const TH1*>> quadPanels;
    for (int q = 1; q <= 4; ++q)
      if (MonitorElement* me = igetter.get(topFolder_ + "/eLinkOccupancyMap_Q" + std::to_string(q)))
        quadPanels.emplace_back("Q" + std::to_string(q), me->getTH1());
    nFiles += TrackerPhase2PlotUtil::savePanelPlots(quadPanels, "eLinkOccupancyMapByQuadrant", plotCfg);

    // Event-averaged occupancy split by detector section, one entry per ELink
    std::vector<std::unique_ptr<TH1F>> sectionHists;
    std::vector<std::pair<std::string, const TH1*>> components;
    for (const auto& [section, color] : kSectionColors) {
      TProfile2D* secProf = sectionMap(igetter, section);
      if (secProf == nullptr)
        continue;
      auto h = bookSectionHist("avg_" + section,
                               avgPSet.getParameter<std::string>("title"),
                               avgPSet.getParameter<int>("NxBins"),
                               avgPSet.getParameter<double>("xmin"),
                               avgPSet.getParameter<double>("xmax"),
                               color);
      for (int ix = 1; ix <= secProf->GetNbinsX(); ix++) {
        for (int iy = 1; iy <= secProf->GetNbinsY(); iy++) {
          if (secProf->GetBinEntries(secProf->GetBin(ix, iy)) > 0)
            h->Fill(secProf->GetBinContent(ix, iy));
        }
      }
      components.emplace_back(section, h.get());
      sectionHists.push_back(std::move(h));
    }
    nFiles += TrackerPhase2PlotUtil::saveStackedPlots(components, "eLinkOccupancyAvgPerSection", plotCfg);

    // Relative difference against a reference run, one entry per ELink
    if (!referenceFile_.empty()) {
      std::unique_ptr<TFile> refFile(TFile::Open(referenceFile_.c_str(), "READ"));
      if (refFile == nullptr || refFile->IsZombie())
        throw cms::Exception("ElinkOccupancyHarvester") << "referenceFile '" << referenceFile_ << "' cannot be opened";

      std::vector<std::unique_ptr<TH1F>> deltaHists;
      std::vector<std::pair<std::string, const TH1*>> deltaComponents;
      for (const auto& [section, color] : kSectionColors) {
        TProfile2D* cur = sectionMap(igetter, section);
        std::unique_ptr<TH1> refHist =
            TrackerPhase2PlotUtil::readHistFromDQMFile(*refFile, topFolder_, "eLinkOccupancyMapPerSection_" + section);
        TProfile2D* ref = dynamic_cast<TProfile2D*>(refHist.get());
        if (cur == nullptr || ref == nullptr)
          continue;
        auto h = bookSectionHist("delta_" + section,
                                 "Relative ELink Occupancy Difference;(this #minus reference) / reference;ELink "
                                 "entries",
                                 deltaBins_,
                                 deltaMin_,
                                 deltaMax_,
                                 color);
        for (int ix = 1; ix <= cur->GetNbinsX(); ix++) {
          for (int iy = 1; iy <= cur->GetNbinsY(); iy++) {
            const int bin = cur->GetBin(ix, iy);
            if (cur->GetBinEntries(bin) <= 0 || ref->GetBinEntries(bin) <= 0 || ref->GetBinContent(bin) <= 0.)
              continue;
            h->Fill((cur->GetBinContent(bin) - ref->GetBinContent(bin)) / ref->GetBinContent(bin));
          }
        }
        deltaComponents.emplace_back(section, h.get());
        deltaHists.push_back(std::move(h));
      }
      if (deltaComponents.empty())
        throw cms::Exception("ElinkOccupancyHarvester")
            << "referenceFile '" << referenceFile_ << "' has no per-section occupancy maps to compare against";
      nFiles += TrackerPhase2PlotUtil::saveStackedPlots(deltaComponents, "eLinkOccupancyDeltaPerSection", plotCfg);
    }

    edm::LogInfo("ElinkOccupancyHarvester") << "wrote " << nFiles << " plot files to " << plotDir_;
  }
}

void ElinkOccupancyHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("TopFolder", "InnerTrackerV/ELink");
  desc.add<std::string>("OccupancyMapName", "eLinkOccupancyMap");
  desc.addUntracked<bool>("savePlots", false);
  desc.addUntracked<std::string>("plotDir", ".");
  desc.addUntracked<std::vector<std::string>>("plotFormats", {"png", "pdf"});
  desc.addUntracked<double>("plotZMax", 1.6);
  desc.addUntracked<std::string>("referenceFile", "");
  desc.addUntracked<int>("deltaBins", 80);
  desc.addUntracked<double>("deltaMin", -0.45);
  desc.addUntracked<double>("deltaMax", 0.15);

  edm::ParameterSetDescription psd0;
  psd0.add<std::string>("name", "eLinkOccupancyAvg");
  psd0.add<std::string>("title", "Event-averaged ELink occupancy;occupancy;ELinks");
  psd0.add<int>("NxBins", 33);
  psd0.add<double>("xmax", 1.6);
  psd0.add<double>("xmin", 0.);
  psd0.add<bool>("switch", true);
  desc.add<edm::ParameterSetDescription>("occupancyAvg", psd0);

  descriptions.add("elinkOccupancyHarvester", desc);
}

DEFINE_FWK_MODULE(ElinkOccupancyHarvester);
