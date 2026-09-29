
// TrackerPhase2HistUtil.h
// -----------------------------------------------------------------------------
// Helper utility for Phase-2 tracker validation & harvesting code:
//  - book 1D histograms from ParameterSets
//  - fill resolution MonitorElements from standard deviations
//  - book numerator/denominator fraction histograms (binomial errors)
//  - book per-event rate histograms (counts / nEvents)
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

  // Books a numerator/denominator fraction (e.g. fake or duplicate fraction)
  // with binomial errors, mirroring TH1::Divide(num, den, 1, 1, "B") in
  // L1Trigger/TrackFindingTracklet/test/L1TrackNtuplePlot.C. Returns the booked
  // ME, or nullptr if an input histogram is missing.
  inline dqm::legacy::MonitorElement* bookFraction(dqm::legacy::DQMStore::IBooker& ibooker,
                                                   dqm::legacy::DQMStore::IGetter& igetter,
                                                   const std::string& numName,
                                                   const std::string& denName,
                                                   const std::string& outName,
                                                   const std::string& title) {
    dqm::legacy::MonitorElement* meNum = igetter.get(numName);
    dqm::legacy::MonitorElement* meDen = igetter.get(denName);
    if (meNum == nullptr || meDen == nullptr) {
      edm::LogWarning("TrackerPhase2HistUtil") << "Missing input histogram(s) for " << outName;
      return nullptr;
    }
    TH1F* hNum = meNum->getTH1F();
    TH1F* hDen = meDen->getTH1F();
    if (hNum == nullptr || hDen == nullptr)
      return nullptr;
    dqm::legacy::MonitorElement* meOut =
        ibooker.book1D(outName, title, hDen->GetNbinsX(), hDen->GetXaxis()->GetXmin(), hDen->GetXaxis()->GetXmax());
    TH1F* hOut = meOut->getTH1F();
    if (hOut->GetSumw2N() == 0)
      hOut->Sumw2();                          // ensure Divide("B") stores the binomial errors
    hOut->Divide(hNum, hDen, 1.0, 1.0, "B");  // "B" = binomial errors
    return meOut;
  }

  inline dqm::legacy::MonitorElement* bookPerEventRate(dqm::legacy::DQMStore::IBooker& ibooker,
                                                       dqm::legacy::DQMStore::IGetter& igetter,
                                                       const std::string& allName,
                                                       const std::string& outName,
                                                       const std::string& title,
                                                       double nEvents) {
    dqm::legacy::MonitorElement* meAll = igetter.get(allName);
    if (meAll == nullptr || nEvents <= 0)
      return nullptr;
    TH1F* hAll = meAll->getTH1F();
    dqm::legacy::MonitorElement* meRate =
        ibooker.book1D(outName, title, hAll->GetNbinsX(), hAll->GetXaxis()->GetXmin(), hAll->GetXaxis()->GetXmax());
    TH1F* hRate = meRate->getTH1F();
    if (hRate->GetSumw2N() == 0)
      hRate->Sumw2();  // so Scale() propagates the Poisson errors
    hRate->Add(hAll);
    hRate->Scale(1.0 / nEvents);
    return meRate;
  }
}  // namespace phase2tkutil
#endif
