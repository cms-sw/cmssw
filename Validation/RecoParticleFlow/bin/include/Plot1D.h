#ifndef PLOT_1D__H
#define PLOT_1D__H

#include "PlotCompareUtility.h"
#include "PlotTypes.h"

#include <TCanvas.h>
#include <TH1F.h>
#include <TLegend.h>
#include <TString.h>
#include <TText.h>

#include <fstream>
#include <iostream>
#include <string>

template <>
inline bool PlotCompareUtility::compare<Plot1D>(HistoData *HD) {
  // get the reference and comparison histograms
  TH1F *href = (TH1F *)HD->getRefHisto();
  TH1F *hnew = (TH1F *)HD->getNewHisto();

  // do not run comparisons if either histogram is empty/broken
  if (hnew == nullptr || href == nullptr || hnew->GetEntries() <= 1 || href->GetEntries() <= 1) {
    // std::cerr << HD->getName() << " error: unable to retrieve histogram (or
    // no entries)\n";
    HD->setIsEmpty(true);
    return false;
  }

  // rebin histograms and center on common range
  if (HD->getDoAllow1DRebinning())
    centerRebin(href, hnew);

  // run statistical comparisons
  double ksScore = hnew->KolmogorovTest(href, "D");
  double chi2Score = hnew->Chi2Test(href, "uup");

  // renormalize histograms for common display
  renormalize(href, hnew);

  // set ks/chi2 and determine high/low scores
  HD->setKSScore(ksScore);
  HD->setChi2Score(chi2Score);
  if (ksThreshold > 0 && chi2Threshold > 0) {
    HD->setLowScore(ksScore < chi2Score ? ksScore : chi2Score);
    HD->setHighScore(ksScore > chi2Score ? ksScore : chi2Score);
  } else if (ksThreshold > 0) {
    HD->setLowScore(ksScore);
    HD->setHighScore(ksScore);
  } else if (chi2Threshold > 0) {
    HD->setLowScore(chi2Score);
    HD->setHighScore(chi2Score);
  } else
    std::cerr << "error: no test performed? chi2Threshold and ksThreshold <= 0\n";

  // check overall result
  bool passed = (ksScore >= ksThreshold && chi2Score >= chi2Threshold);
  HD->setResult(passed);
  //  if (!passed) std::cout << "NOTPASSED!!!!"; else std::cout << "YESPASSED!";
  // returns true on test passed and false on test failed
  HD->setIsEmpty(false);
  return passed;
}

template <>
inline void PlotCompareUtility::makePlots<Plot1D>(HistoData *HD) {
  std::cerr << HD->getName() << "makePlots<Plot1D>\n";

  // do not make any new plot if empty
  if (HD->getIsEmpty()) {
    HD->setResultImage("NoData_Results.gif");
    HD->setResultTarget("NoData_Results.gif");
    return;
  }

  // get the reference and comparison histograms
  TH1F *href = (TH1F *)HD->getRefHisto();
  TH1F *hnew = (TH1F *)HD->getNewHisto();

  // set drawing options on the reference histogram
  href->SetStats(false);
  href->SetLineWidth(2);
  href->SetLineColor(14);
  href->SetMarkerColor(14);
  href->SetFillColor(18);

  // set drawing options on the new histogram
  hnew->SetStats(false);
  hnew->SetLineWidth(2);
  hnew->SetLineColor(HD->getShadedLineColor());
  hnew->SetFillStyle(HD->getShadedFillStyle());
  hnew->SetFillColor(HD->getShadedFillColor());

  // place the test results as the title
  TString title = HD->getName();
  if (ksThreshold > 0) {
    title += " KS Score = ";
    title += HD->getKSScore();
  }
  if (chi2Threshold > 0) {
    title += " Chi^2 Score = ";
    title += HD->getChi2Score();
  }

  // the canvas is rescaled during gif conversion, so add padding to Canvas
  // dimensions
  int plotsCanvasWidth = plotsWidth + 4;
  int plotsCanvasHeight = plotsHeight + 28;

  // setup canvas for displaying the compared histograms
  TCanvas hCanvas("hCanvas", "hCanvas", plotsCanvasWidth, plotsCanvasHeight);
  hCanvas.SetTopMargin(float(plotsTopMargin) / plotsHeight);
  hCanvas.SetLeftMargin(float(plotsLeftMargin) / plotsWidth);
  hCanvas.SetRightMargin(float(plotsRightMargin) / plotsWidth);
  hCanvas.SetBottomMargin(float(plotsBottomMargin) / plotsHeight);
  hCanvas.SetFrameFillColor(10);
  hCanvas.SetGrid();
  hCanvas.SetLogy(1);
  hCanvas.Draw();

  TText canvasTitle(0.1, 0.97, title.Data());
  canvasTitle.Draw("SAME");

  // draw the histograms
  href->Draw();
  hnew->Draw("SAME");
  if (HD->getDoDrawErrorBars())
    hnew->Draw("E1SAME");

  // draw a legend
  TLegend legend(0.15, 0.01, 0.3, 0.08);
  legend.AddEntry(hnew, "New", "lF");
  legend.AddEntry(href, "Reference", "lF");
  legend.SetFillColor(kNone);
  legend.Draw("SAME");

  // create the plots overlay image
  std::string gifName = HD->getName() + "_Results.gif";
  if (HD->getResultImage().empty())
    HD->setResultImage(gifName);
  if (HD->getResultTarget().empty())
    HD->setResultTarget(gifName);
  std::cerr << "About to print" << gifName << "\n";
  hCanvas.Print(gifName.c_str());
}

template <>
inline void PlotCompareUtility::makeHTML<Plot1D>(HistoData *HD) {
  /* HTML is not presently required for 1D histograms -- do nothing

  // create HTML support code for this HistoData
  std::string Name = hd->getName();
  std::string gifName = Name + "_Results.gif";
  std::string html = Name + "_Results.html";
  ofstream fout(html.c_str());

  // simply link to the appropriate image overlay
  fout << "<html><body><img src=\"" << gifName << "\"></body></html>" << endl;

  // close the file
  fout.close();

  */
}

#endif  // PLOT_1D__H
