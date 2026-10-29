// -*- C++ -*-
// Package:    Validation/SiTrackerPhase2V
// Class:      SlinkOccupancyHarvester
// Description: Harvest the per-slink occupancy DQM
//
// Author: Lacey Dishman, Sihyun Jeon (Boston University)
// Written: August 2026

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2HarvestingUtil.h"
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2PlotUtil.h"

#include "TColor.h"
#include "TH1F.h"
#include "TProfile2D.h"

class SlinkOccupancyHarvester : public DQMEDHarvester {
public:
  explicit SlinkOccupancyHarvester(const edm::ParameterSet&);
  ~SlinkOccupancyHarvester() override;
  void dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::ParameterSet config_;
  const std::string topFolder_;
  const std::string occupancyMapName_;
  const bool savePlots_;
  const std::string plotDir_;
  const std::vector<std::string> plotFormats_;
  const double zMax_;
};

SlinkOccupancyHarvester::SlinkOccupancyHarvester(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      topFolder_(iConfig.getParameter<std::string>("TopFolder")),
      occupancyMapName_(iConfig.getParameter<std::string>("OccupancyMapName")),
      savePlots_(iConfig.getUntrackedParameter<bool>("savePlots", false)),
      plotDir_(iConfig.getUntrackedParameter<std::string>("plotDir", ".")),
      plotFormats_(iConfig.getUntrackedParameter<std::vector<std::string>>("plotFormats", {"png", "pdf"})),
      zMax_(iConfig.getUntrackedParameter<double>("plotZMax", 1.6)) {}

SlinkOccupancyHarvester::~SlinkOccupancyHarvester() {}

void SlinkOccupancyHarvester::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  MonitorElement* occMap = igetter.get(topFolder_ + "/" + occupancyMapName_);
  if (occMap == nullptr) {
    edm::LogWarning("SlinkOccupancyHarvester") << occupancyMapName_ << " not found in " << topFolder_;
    return;
  }
  TProfile2D* prof = occMap->getTProfile2D();

  MonitorElement* nEvtME = igetter.get(topFolder_ + "/nEvents");
  const double nEvents = (nEvtME != nullptr) ? nEvtME->getTH1F()->GetBinContent(1) : 0.;
  if (nEvents <= 0) {
    edm::LogWarning("SlinkOccupancyHarvester")
        << "nEvents missing or empty in " << topFolder_ << "; skipping all normalization.";
  }

  // Normalize the raw 1D occupancy to per-event
  MonitorElement* slinkOcc = igetter.get(topFolder_ + "/slinkOccupancy");
  if (slinkOcc != nullptr && nEvents > 0) {
    slinkOcc->getTH1F()->Scale(1.0 / nEvents);
    slinkOcc->getTH1F()->SetOption("HIST");
    slinkOcc->setAxisTitle("SLink entries / nevents", 2);
  }

  // Normalization for per-DTC full spectrum occupancy
  for (auto* me : igetter.getAllContents(topFolder_)) {
    if (me == nullptr)
      continue;
    if (me->getName().rfind("slinkSpectrumOccupancyPerDTC_", 0) == 0 && nEvents > 0) {
      me->getTH1F()->Scale(1.0 / nEvents);
      me->getTH1F()->SetOption("HIST");
      me->setAxisTitle("SLink entries / nevents", 2);
    }
  }

  MonitorElement* occVsDTC = igetter.get(topFolder_ + "/slinkOccupancyVsDTC");
  if (occVsDTC != nullptr && nEvents > 0) {
    occVsDTC->getTH2F()->Scale(1.0 / nEvents);
    occVsDTC->getTH2F()->SetOption("COLZ");
    occVsDTC->setAxisTitle("SLink entries / nevents", 3);
  }

  // Event-averaged 1D occupancy, one entry per SLink
  ibooker.cd();
  ibooker.setCurrentFolder(topFolder_);
  MonitorElement* occAvg =
      phase2tkharvestutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("occupancyAvg"), ibooker);
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
    plotCfg.occMapNamePrefix = "slinkOccupancyMap";
    plotCfg.zMax = zMax_;
    plotCfg.gaussFitNamePrefix = "dataSizePerDTC";
    int nFiles = TrackerPhase2PlotUtil::saveFolderPlots(igetter, topFolder_, plotCfg);

    // Event-averaged occupancy per DTC, one entry per SLink, stacked per quadrant (DTC 11-19, ..., 41-49)
    std::vector<std::unique_ptr<TH1F>> dtcHists;
    std::map<int, std::vector<std::pair<std::string, TH1F*>>> byQuad;
    const TH1F* avgHist = occAvg->getTH1F();
    for (int ix = 1; ix <= prof->GetNbinsX(); ix++) {
      const std::string dtc = prof->GetXaxis()->GetBinLabel(ix);
      if (dtc.empty())
        continue;
      const int decade = std::stoi(dtc) / 10;
      const std::string q = std::to_string(decade);
      auto h =
          std::make_unique<TH1F>(("avg_dtc" + dtc).c_str(),
                                 ("Event-Averaged SLink Occupancy, Q" + q + " (DTC " + q + "1-" + q + "9)").c_str(),
                                 avgHist->GetNbinsX(),
                                 avgHist->GetXaxis()->GetXmin(),
                                 avgHist->GetXaxis()->GetXmax());
      h->SetDirectory(nullptr);
      h->GetXaxis()->SetTitle(avgHist->GetXaxis()->GetTitle());
      h->GetYaxis()->SetTitle(avgHist->GetYaxis()->GetTitle());
      for (int iy = 1; iy <= prof->GetNbinsY(); iy++) {
        if (prof->GetBinEntries(prof->GetBin(ix, iy)) > 0)
          h->Fill(prof->GetBinContent(ix, iy));
      }
      byQuad[decade].emplace_back("DTC " + dtc, h.get());
      dtcHists.push_back(std::move(h));
    }
    for (const auto& [decade, hists] : byQuad) {
      std::vector<std::pair<std::string, const TH1*>> components;
      for (size_t i = 0; i < hists.size(); ++i) {
        // spread the quadrant's DTCs over the palette so neighbours stay distinguishable
        hists[i].second->SetLineColor(TColor::GetColorPalette(
            static_cast<int>(i * (TColor::GetNumberOfColors() - 1) / std::max<size_t>(1, hists.size() - 1))));
        components.emplace_back(hists[i].first, hists[i].second);
      }
      nFiles += TrackerPhase2PlotUtil::saveStackedPlots(
          components, "slinkOccupancyAvgPerDTC_Q" + std::to_string(decade), plotCfg);
    }

    edm::LogInfo("SlinkOccupancyHarvester") << "wrote " << nFiles << " plot files to " << plotDir_;
  }
}

void SlinkOccupancyHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("TopFolder", "InnerTrackerV/SLink");
  desc.add<std::string>("OccupancyMapName", "slinkOccupancyMap");
  desc.addUntracked<bool>("savePlots", false);
  desc.addUntracked<std::string>("plotDir", ".");
  desc.addUntracked<std::vector<std::string>>("plotFormats", {"png", "pdf"});
  desc.addUntracked<double>("plotZMax", 1.6);

  edm::ParameterSetDescription psd0;
  psd0.add<std::string>("name", "slinkOccupancyAvg");
  psd0.add<std::string>("title", "Event-averaged SLink occupancy;occupancy;SLinks");
  psd0.add<int>("NxBins", 33);
  psd0.add<double>("xmax", 1.6);
  psd0.add<double>("xmin", 0.);
  psd0.add<bool>("switch", true);
  desc.add<edm::ParameterSetDescription>("occupancyAvg", psd0);

  descriptions.add("slinkOccupancyHarvester", desc);
}

DEFINE_FWK_MODULE(SlinkOccupancyHarvester);
