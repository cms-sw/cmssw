// -*- C++ -*-
// Package:    Validation/SiTrackerPhase2V
// Class:      TrackerPhase2PlotUtil
// Description: Renders the MonitorElements of one DQM folder to image files
//
// Author: Lacey Dishman, Sihyun Jeon (Boston University)
// Written: August 2026

#ifndef Validation_SiTrackerPhase2V_TrackerPhase2PlotUtil_H
#define Validation_SiTrackerPhase2V_TrackerPhase2PlotUtil_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"

class TFile;
class TH1;

namespace TrackerPhase2PlotUtil {

  struct PlotConfig {
    std::string plotDir = ".";
    std::vector<std::string> formats = {"png", "pdf"};
    // MEs with this prefix get the occupancy-map rendering
    std::string occMapNamePrefix;
    double zMax = 1.6;
    // 1D MEs with this prefix get a Gaussian fit drawn on top
    std::string gaussFitNamePrefix;
  };

  int saveFolderPlots(dqm::harvesting::DQMStore::IGetter& igetter, const std::string& folder, const PlotConfig& cfg);

  // Renders components as '<baseName>Stacked' and '<baseName>Overlaid'
  int saveStackedPlots(const std::vector<std::pair<std::string, const TH1*>>& components,
                       const std::string& baseName,
                       const PlotConfig& cfg);

  // Renders the 2D maps as one figure '<name>', one pad per panel stacked vertically
  int savePanelPlots(const std::vector<std::pair<std::string, const TH1*>>& panels,
                     const std::string& name,
                     const PlotConfig& cfg);

  std::unique_ptr<TH1> readHistFromDQMFile(TFile& file, const std::string& folder, const std::string& name);

}  // namespace TrackerPhase2PlotUtil

#endif  // Validation_SiTrackerPhase2V_TrackerPhase2PlotUtil_H
