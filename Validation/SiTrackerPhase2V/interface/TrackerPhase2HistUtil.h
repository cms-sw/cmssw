
// TrackerPhase2HistUtil.h
// -----------------------------------------------------------------------------
// Helper utility for booking 1D histograms from ParameterSets and filling
// resolution MonitorElements (using standard deviations) in Phase-2 tracker
// validation & harvesting code.
//
// Author: Brandi Skipworth
// -----------------------------------------------------------------------------

#ifndef Validation_SiTrackerPhase2V_TrackerPhase2HistUtil_h
#define Validation_SiTrackerPhase2V_TrackerPhase2HistUtil_h

#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TH1.h"

namespace phase2tkutil {

  inline void fillResolutionFromVec(const std::vector<dqm::legacy::MonitorElement*>& srcVec,
                                    dqm::legacy::MonitorElement* destME,
                                    const std::string& yAxisTitle) {
    if (!destME)
      return;

    // Check if any source elements are missing
    if (std::find(srcVec.begin(), srcVec.end(), nullptr) != srcVec.end()) {
      edm::LogWarning("TrackerPhase2HistUtil") << "Missing source ME for resolution: " << destME->getName();
      return;
    }

    TH1* hDest = destME->getTH1();
    hDest->SetMinimum(0.0);
    hDest->SetStats(false);
    if (!yAxisTitle.empty()) {
      hDest->GetYaxis()->SetTitle(yAxisTitle.c_str());
    }

    for (size_t i = 0; i < srcVec.size(); ++i) {
      TH1* hSrc = srcVec[i]->getTH1();
      // Bin 1 in destination corresponds to index 0 in vector
      hDest->SetBinContent(i + 1, hSrc->GetStdDev());
      hDest->SetBinError(i + 1, hSrc->GetStdDevError());
    }
  }

  inline dqm::legacy::MonitorElement* book1DFromPS(dqm::legacy::DQMStore::IBooker& iBooker,
                                                   const std::string& name,
                                                   const edm::ParameterSet& ps,
                                                   const char* xaxis,
                                                   const char* yaxis) {
    auto h = iBooker.book1D(name,
                            name,
                            ps.getParameter<int32_t>("Nbinsx"),
                            ps.getParameter<double>("xmin"),
                            ps.getParameter<double>("xmax"));
    h->setAxisTitle(xaxis, 1);
    h->setAxisTitle(yaxis, 2);
    return h;
  }
}  // namespace phase2tkutil
#endif
