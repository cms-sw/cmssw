#include "include/HistoData.h"

#include <iostream>

#include <TFile.h>
#include <TH1.h>
#include <TLine.h>
#include <TNamed.h>
#include <TPave.h>

using namespace std;

HistoData::HistoData(
    std::string Name, int Type, int Bin, string NewPath, TFile *NewFile, string RefPath, TFile *RefFile) {
  name = Name;
  type = Type;
  bin = Bin;

  newHisto = dynamic_cast<TH1 *>(NewFile->Get(NewPath.c_str()));
  refHisto = dynamic_cast<TH1 *>(RefFile->Get(RefPath.c_str()));

  initialize();
}

HistoData::HistoData(std::string Name, int Type, int Bin, TH1 *NewHisto, TH1 *RefHisto) {
  name = Name;
  type = Type;
  bin = Bin;
  newHisto = NewHisto;
  refHisto = RefHisto;

  initialize();
}

void HistoData::initialize() {
  // scores/tests initialization
  lowScore = 10.0;
  highScore = 0.0;
  ksScore = 0.0;
  chi2Score = 0.0;
  result = true;
  isEmpty = true;

  // for HTML display
  // resultImage = "NoData_Results.gif";
  // resultTarget = "NoData_Results.gif";
  resultImage = "";
  resultTarget = "";

  // rebinning/projections, etc.
  doDrawErrorBars = false;
  doAllow1DRebinning = false;
  doAllow2DRebinningX = true;
  doAllow2DRebinningY = true;
  doProjectionsX = false;
  doProjectionsY = true;
  maxProjectionsX = 20;
  maxProjectionsY = 20;

  // test result color conventions
  passColor = kGreen;
  failColor = kRed;
  errorColor = 16;  // kGray

  // default color and style scheme
  lineUseFillColor = true;
  solidLineColor = kWhite;
  solidFillColor = kWhite;
  solidFillStyle = 1001;
  shadedLineColor = errorColor;
  shadedFillColor = errorColor;
  shadedFillStyle = 3001;
}

void HistoData::setResult(bool Result) {
  // set the test result
  result = Result;

  // set the color scheme
  solidFillColor = result ? passColor : failColor;
  shadedFillColor = result ? passColor : failColor;
}

void HistoData::dump() {
  cout << "name      = " << name << endl
       << "type      = " << type << endl
       << "bin       = " << bin << endl
       << "ksScore   = " << ksScore << endl
       << "chi2Score = " << chi2Score << endl
       << "result    = " << (result ? "pass" : "fail") << endl;
}

void HistoData::drawResult(TH1 *Summary, bool Vertical, bool SetBinLabel) {
  // add label to the summary if desired
  if (SetBinLabel) {
    Summary->GetXaxis()->SetBinLabel(bin, getRefHisto()->GetTitle());
    // Summary->GetXaxis()->SetBinLabel(bin,name.c_str());
  } else
    Summary->GetXaxis()->SetBinLabel(bin, name.c_str());

  double minimum = Summary->GetMinimum();
  // determine where to draw the result (score axis)
  //   1: solid bar starts
  //   2: solid bar ends, hatched bar starts
  //   3: hatched bar ends
  double score1 = minimum;
  double score2 = (lowScore == 10. || lowScore < minimum) ? minimum : lowScore;
  double score3 = (lowScore == 10.) ? 1 : ((highScore < minimum) ? minimum : highScore);

  // determine where to draw the result (binning axis)
  double binCenter = Summary->GetBinCenter(bin);
  double binWidth = Summary->GetBinWidth(bin);
  double bin1 = binCenter - binWidth / 3;
  double bin2 = binCenter + binWidth / 3;

  // set coordinates of display based on plot alignment
  double solidX1, solidY1, solidX2, solidY2;
  double hatchedX1, hatchedY1, hatchedX2, hatchedY2;
  double axisX1, axisY1, axisX2, axisY2;
  if (Vertical) {
    solidX1 = bin1;
    solidX2 = bin2;
    solidY1 = score1;
    solidY2 = score2;
    hatchedX1 = bin1;
    hatchedX2 = bin2;
    hatchedY1 = score2;
    hatchedY2 = score3;
    axisX1 = bin1;
    axisX2 = bin2;
    axisY1 = minimum;
    axisY2 = minimum;
  } else {
    solidX1 = score1;
    solidX2 = score2;
    solidY1 = bin1;
    solidY2 = bin2;
    hatchedX1 = score2;
    hatchedX2 = score3;
    hatchedY1 = bin1;
    hatchedY2 = bin2;
    axisX1 = minimum;
    axisX2 = minimum;
    axisY1 = bin1;
    axisY2 = bin2;
  }

  // a solid bar is drawn from zero to the lowest score
  if (lowScore > minimum && lowScore != 10.) {
    TPave *solidBar = new TPave(solidX1, solidY1, solidX2, solidY2, 1, "");
    solidBar->SetBit(kCanDelete);
    solidBar->SetLineColor(lineUseFillColor ? solidFillColor : solidLineColor);
    solidBar->SetFillColor(solidFillColor);
    solidBar->SetFillStyle(solidFillStyle);
    solidBar->Draw();
  }

  // a hatched bar is drawn from the lowest score to the highest score
  if ((lowScore != highScore && highScore > minimum) || lowScore == 10.) {
    TPave *hatchedBar = new TPave(hatchedX1, hatchedY1, hatchedX2, hatchedY2, 1, "");
    hatchedBar->SetBit(kCanDelete);
    hatchedBar->SetLineColor(lineUseFillColor ? shadedFillColor : shadedLineColor);
    hatchedBar->SetFillColor(shadedFillColor);
    hatchedBar->SetFillStyle(3004);
    hatchedBar->Draw();
  }

  // paste a line over the base axis to fix border color artifacts
  TLine *axisLine = new TLine(axisX1, axisY1, axisX2, axisY2);
  if (Vertical)
    axisLine->SetLineColor(Summary->GetAxisColor("X"));
  else
    axisLine->SetLineColor(Summary->GetAxisColor("Y"));
  axisLine->SetBit(kCanDelete);
  axisLine->Draw("SAME");

  // paste a line before (in this proceeding that means after) Barrel and Endcap
  // Plots
  if (name == "ERPt" || name == "BRPt") {
    axisY1 = axisY2 = binCenter + binWidth / 2;
    axisX2 = Summary->GetMaximum();
    axisX1 = Summary->GetMinimum() - 200;
    TLine *regionLine = new TLine(axisX1, axisY1, axisX2, axisY2);
    regionLine->SetLineColor(Summary->GetAxisColor("X"));
    regionLine->SetBit(kCanDelete);
    regionLine->SetLineWidth(3);
    regionLine->Draw();
  }
}
