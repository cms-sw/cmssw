// -*- C++ -*-
// Package:    Validation/SiTrackerPhase2V
// Class:      TrackerPhase2PlotUtil
// Description: Renders the MonitorElements of one DQM folder to image files
//
// Author: Lacey Dishman, Sihyun Jeon (Boston University)
// Written: August 2026

#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2PlotUtil.h"

#include <algorithm>
#include <filesystem>
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TProfile2D.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TF1.h"
#include "TPad.h"
#include "TFile.h"
#include "TH1.h"
#include "THStack.h"
#include "TKey.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TStyle.h"

namespace {

  constexpr int kStdWidthPx = 900;
  constexpr int kStdHeightPx = 600;
  constexpr int kWideMinBins = 40;
  constexpr int kWidePxPerBin = 11;
  constexpr int kWidePadPx = 300;
  constexpr int kWideMinPx = 760;
  constexpr int kWideMaxPx = 4200;
  constexpr int kWideHeightPx = 400;

  class StyleGuard {
  public:
    StyleGuard() : optStat_(gStyle->GetOptStat()), nContours_(gStyle->GetNumberContours()) {
      const TArrayI& pal = TColor::GetPalette();
      palette_.assign(pal.GetArray(), pal.GetArray() + pal.GetSize());
    }
    ~StyleGuard() {
      gStyle->SetOptStat(optStat_);
      gStyle->SetNumberContours(nContours_);
      if (!palette_.empty())
        gStyle->SetPalette(static_cast<Int_t>(palette_.size()), palette_.data());
    }
    StyleGuard(const StyleGuard&) = delete;
    StyleGuard& operator=(const StyleGuard&) = delete;

  private:
    int optStat_;
    int nContours_;
    std::vector<Int_t> palette_;
  };

  // drop empty rows at the top, e.g. the 6-row ELink map when no module uses all of them
  void cropEmptyRows(TH2* h2) {
    auto* prof = dynamic_cast<TProfile2D*>(h2);
    int top = 0;
    for (int iy = h2->GetNbinsY(); iy >= 1 && top == 0; --iy)
      for (int ix = 1; ix <= h2->GetNbinsX() && top == 0; ++ix) {
        const int bin = h2->GetBin(ix, iy);
        if ((prof ? prof->GetBinEntries(bin) : h2->GetBinContent(bin)) != 0)
          top = iy;
      }
    if (top > 0 && top < h2->GetNbinsY())
      h2->GetYaxis()->SetRange(1, top);
  }

}  // namespace

int TrackerPhase2PlotUtil::saveFolderPlots(dqm::harvesting::DQMStore::IGetter& igetter,
                                           const std::string& folder,
                                           const PlotConfig& cfg) {
  std::error_code ec;
  std::filesystem::create_directories(cfg.plotDir, ec);
  if (ec) {
    edm::LogWarning("TrackerPhase2PlotUtil")
        << "cannot create plot directory '" << cfg.plotDir << "': " << ec.message() << "; no plots written";
    return 0;
  }

  StyleGuard styleGuard;
  gStyle->SetOptStat(0);

  int nWritten = 0;
  for (dqm::harvesting::MonitorElement* me : igetter.getAllContents(folder)) {
    if (me == nullptr)
      continue;

    bool is2D = false;
    switch (me->kind()) {
      case dqm::harvesting::MonitorElement::Kind::TH1F:
      case dqm::harvesting::MonitorElement::Kind::TH1D:
      case dqm::harvesting::MonitorElement::Kind::TPROFILE:
        break;
      case dqm::harvesting::MonitorElement::Kind::TH2F:
      case dqm::harvesting::MonitorElement::Kind::TH2D:
      case dqm::harvesting::MonitorElement::Kind::TPROFILE2D:
        is2D = true;
        break;
      default:
        continue;
    }
    TH1* stored = me->getTH1();
    if (stored == nullptr || stored->GetEntries() == 0)
      continue;

    std::unique_ptr<TH1> h(static_cast<TH1*>(stored->Clone()));
    h->SetDirectory(nullptr);

    const std::string& name = me->getName();
    const bool isOccMap = is2D && !cfg.occMapNamePrefix.empty() && name.rfind(cfg.occMapNamePrefix, 0) == 0;

    if (isOccMap) {
      gStyle->SetPalette(kRainBow);
      h->SetMinimum(0.);
      h->SetMaximum(cfg.zMax);
    } else {
      gStyle->SetPalette(kViridis);
    }

    const int nx = h->GetNbinsX();
    const bool wide = isOccMap && nx > kWideMinBins;
    const int width = wide ? std::clamp(nx * kWidePxPerBin + kWidePadPx, kWideMinPx, kWideMaxPx) : kStdWidthPx;
    const int height = wide ? kWideHeightPx : kStdHeightPx;

    TCanvas canvas(("c_" + name).c_str(), name.c_str(), width, height);
    if (isOccMap) {
      canvas.SetLeftMargin(0.075);
      canvas.SetRightMargin(0.115);
      canvas.SetTopMargin(0.13);
      canvas.SetBottomMargin(0.16);
    } else {
      canvas.SetLeftMargin(0.11);
      canvas.SetRightMargin(is2D ? 0.135 : 0.05);
      canvas.SetTopMargin(0.10);
      canvas.SetBottomMargin(0.13);
    }

    if (is2D) {
      cropEmptyRows(static_cast<TH2*>(h.get()));
      if (isOccMap && h->GetXaxis()->GetLabels() != nullptr) {
        h->GetXaxis()->SetLabelSize(0.040);
        h->GetXaxis()->LabelsOption("h");
      }
      h->Draw("COLZ");
    } else if (me->kind() == dqm::harvesting::MonitorElement::Kind::TPROFILE) {
      h->SetLineWidth(2);
      h->SetMarkerStyle(20);
      h->SetMarkerSize(0.9);
      h->Draw("E1");
    } else {
      h->SetLineWidth(2);
      h->SetFillColorAlpha(kAzure + 1, 0.35);
      h->Draw("HIST");
      if (!cfg.gaussFitNamePrefix.empty() && name.rfind(cfg.gaussFitNamePrefix, 0) == 0 && h->GetEntries() > 0) {
        TF1 fit("gausFit", "gaus", h->GetMean() - 4 * h->GetRMS(), h->GetMean() + 4 * h->GetRMS());
        if (h->Fit(&fit, "QNR") == 0) {
          fit.SetLineColor(kRed);
          fit.SetLineWidth(2);
          fit.DrawCopy("SAME");
          TPaveText box(0.60, 0.72, 0.92, 0.87, "NDC");
          box.SetBorderSize(1);
          box.SetFillColor(0);
          box.SetTextAlign(12);
          box.AddText(Form("Mean: %.2f", fit.GetParameter(1)));
          box.AddText(Form("Std Dev: %.2f", std::abs(fit.GetParameter(2))));
          box.DrawClone();
        }
      }
    }

    for (const auto& fmt : cfg.formats) {
      if (fmt.empty())
        continue;
      canvas.SaveAs((cfg.plotDir + "/" + name + "." + fmt).c_str());
      ++nWritten;
    }
  }
  return nWritten;
}

int TrackerPhase2PlotUtil::savePanelPlots(const std::vector<std::pair<std::string, const TH1*>>& panels,
                                          const std::string& name,
                                          const PlotConfig& cfg) {
  if (panels.empty())
    return 0;
  std::error_code ec;
  std::filesystem::create_directories(cfg.plotDir, ec);
  if (ec)
    return 0;

  StyleGuard styleGuard;
  gStyle->SetOptStat(0);
  gStyle->SetPalette(kRainBow);

  int maxNx = 1;
  for (const auto& [label, hist] : panels)
    maxNx = std::max(maxNx, hist->GetNbinsX());
  const int n = static_cast<int>(panels.size());
  const int width = std::clamp(maxNx * kWidePxPerBin + kWidePadPx, kWideMinPx, kWideMaxPx);
  TCanvas canvas(("c_" + name).c_str(), name.c_str(), width, n * kWideHeightPx);
  canvas.Divide(1, n, 0., 0.);

  std::vector<std::unique_ptr<TH1>> clones;
  for (int i = 0; i < n; ++i) {
    canvas.cd(i + 1);
    gPad->SetLeftMargin(0.075);
    gPad->SetRightMargin(0.115);
    gPad->SetTopMargin(0.13);
    gPad->SetBottomMargin(0.16);
    std::unique_ptr<TH1> h(static_cast<TH1*>(panels[i].second->Clone()));
    h->SetDirectory(nullptr);
    if (!panels[i].first.empty())
      h->SetTitle(
          (std::string(panels[i].second->GetTitle()).substr(0, std::string(panels[i].second->GetTitle()).find(';')))
              .c_str());
    h->SetMinimum(0.);
    h->SetMaximum(cfg.zMax);
    cropEmptyRows(static_cast<TH2*>(h.get()));
    h->Draw("COLZ");
    clones.push_back(std::move(h));
  }

  int nWritten = 0;
  for (const auto& fmt : cfg.formats) {
    if (fmt.empty())
      continue;
    canvas.SaveAs((cfg.plotDir + "/" + name + "." + fmt).c_str());
    ++nWritten;
  }
  return nWritten;
}

std::unique_ptr<TH1> TrackerPhase2PlotUtil::readHistFromDQMFile(TFile& file,
                                                                const std::string& folder,
                                                                const std::string& name) {
  // '<Top>/<sub>/<me>' is stored as 'DQMData/Run N/<Top>/Run summary/<sub>/<me>'
  const size_t slash = folder.find('/');
  const std::string top = folder.substr(0, slash);
  const std::string rest = (slash == std::string::npos) ? "" : "/" + folder.substr(slash + 1);

  TDirectory* dqmData = file.GetDirectory("DQMData");
  if (dqmData == nullptr)
    return nullptr;

  for (const auto&& key : *dqmData->GetListOfKeys()) {
    const std::string run = key->GetName();
    if (run.rfind("Run ", 0) != 0)
      continue;
    const std::string path = "DQMData/" + run + "/" + top + "/Run summary" + rest + "/" + name;
    if (TH1* h = dynamic_cast<TH1*>(file.Get(path.c_str()))) {
      std::unique_ptr<TH1> clone(static_cast<TH1*>(h->Clone()));
      clone->SetDirectory(nullptr);
      return clone;
    }
  }
  return nullptr;
}

int TrackerPhase2PlotUtil::saveStackedPlots(const std::vector<std::pair<std::string, const TH1*>>& components,
                                            const std::string& baseName,
                                            const PlotConfig& cfg) {
  if (components.empty())
    return 0;

  std::error_code ec;
  std::filesystem::create_directories(cfg.plotDir, ec);
  if (ec) {
    edm::LogWarning("TrackerPhase2PlotUtil")
        << "cannot create plot directory '" << cfg.plotDir << "': " << ec.message() << "; no plots written";
    return 0;
  }

  StyleGuard styleGuard;
  gStyle->SetOptStat(0);

  const TH1* first = components.front().second;
  const std::string axisTitles = std::string(";") + first->GetXaxis()->GetTitle() + ";" + first->GetYaxis()->GetTitle();

  // Spread long component lists over several columns
  const int nColumns = 1 + static_cast<int>(components.size() - 1) / 13;
  auto makeLegend = [&]() {
    auto legend = std::make_unique<TLegend>(0.93 - 0.13 * nColumns, 0.52, 0.93, 0.88);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->SetNColumns(nColumns);
    legend->SetTextSize(nColumns > 1 ? 0.020 : 0.028);
    return legend;
  };
  auto saveCanvas = [&](TCanvas& canvas, const std::string& suffix) {
    int n = 0;
    for (const auto& fmt : cfg.formats) {
      if (fmt.empty())
        continue;
      canvas.SaveAs((cfg.plotDir + "/" + baseName + suffix + "." + fmt).c_str());
      ++n;
    }
    return n;
  };

  int nWritten = 0;
  std::vector<std::unique_ptr<TH1>> clones;
  auto cloneComponents = [&]() {
    clones.clear();
    for (const auto& [label, hist] : components) {
      std::unique_ptr<TH1> h(static_cast<TH1*>(hist->Clone()));
      h->SetDirectory(nullptr);
      clones.push_back(std::move(h));
    }
  };

  {
    TCanvas canvas("c_stacked", "", kStdWidthPx, kStdHeightPx);
    canvas.SetLeftMargin(0.11);
    canvas.SetRightMargin(0.05);
    canvas.SetTopMargin(0.10);
    canvas.SetBottomMargin(0.13);
    THStack stack("stack", (std::string(first->GetTitle()) + axisTitles).c_str());
    auto legend = makeLegend();
    cloneComponents();
    for (size_t i = 0; i < clones.size(); ++i) {
      TH1* h = clones[i].get();
      h->SetFillColorAlpha(h->GetLineColor(), 0.8);
      h->SetLineWidth(1);
      stack.Add(h, "HIST");
      legend->AddEntry(h, components[i].first.c_str(), "f");
    }
    stack.SetMaximum(1.25 * stack.GetMaximum());
    stack.Draw("HIST");
    legend->Draw();
    canvas.RedrawAxis();
    nWritten += saveCanvas(canvas, "Stacked");
  }

  {
    TCanvas canvas("c_overlaid", "", kStdWidthPx, kStdHeightPx);
    canvas.SetLeftMargin(0.11);
    canvas.SetRightMargin(0.05);
    canvas.SetTopMargin(0.10);
    canvas.SetBottomMargin(0.13);
    auto legend = makeLegend();
    cloneComponents();
    double yMax = 0.;
    for (const auto& h : clones)
      yMax = std::max(yMax, h->GetMaximum());
    // a distinct marker per component keeps overlapping lines attributable
    static const int kMarkers[] = {20, 21, 22, 23, 29, 33, 34, 39, 41, 43, 45, 47, 49};
    for (size_t i = 0; i < clones.size(); ++i) {
      TH1* h = clones[i].get();
      h->SetFillStyle(0);
      h->SetLineWidth(2);
      h->SetMarkerStyle(kMarkers[i % 13]);
      h->SetMarkerColor(h->GetLineColor());
      h->SetMarkerSize(1.1);
      if (i == 0) {
        h->SetTitle((std::string(first->GetTitle()) + axisTitles).c_str());
        h->SetMaximum(1.25 * yMax);
        h->Draw("HIST");
      } else {
        h->Draw("HIST SAME");
      }
      h->Draw("P SAME");
      legend->AddEntry(h, components[i].first.c_str(), "lp");
    }
    legend->Draw();
    canvas.RedrawAxis();
    nWritten += saveCanvas(canvas, "Overlaid");
  }

  return nWritten;
}
