#include "include/PlotCompareUtility.h"

#include "include/Plot1D.h"
#include "include/Plot2D.h"
#include "include/PlotTypes.h"
#include "include/Style.h"

#include <fstream>

#include <TCanvas.h>
#include <TFile.h>
#include <TGaxis.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLine.h>
#include <TPave.h>
#include <TStyle.h>
#include <TText.h>
using namespace std;

PlotCompareUtility::PlotCompareUtility(std::string Reference,
                                       std::string New,
                                       std::string NewBasePath,
                                       std::string NewPrefix,
                                       std::string RefBasePath,
                                       std::string RefPrefix) {
  // open TFiles
  cout << refFile << " " << newFile << endl;
  refFile = new TFile(Reference.c_str(), "READ");
  newFile = new TFile(New.c_str(), "READ");

  // set the data path and prefix
  newBasePath = NewBasePath;
  newPrefix = NewPrefix;
  refBasePath = RefBasePath;
  refPrefix = RefPrefix;

  // set default thresholds
  ksThreshold = 0;
  chi2Threshold = 0;

  // overall summary of results plot
  summaryWidth = 700;
  summaryBarsThickness = 20;
  summaryTopMargin = 60;
  summaryLeftMargin = 250;  // former 180
  summaryRightMargin = 60;
  summaryBottomMargin = 60;

  // summary of results for 2d projections
  projectionsHeight = 340;
  projectionsBarsThickness = 20;
  projectionsTopMargin = 20;
  projectionsBottomMargin = 60;
  projectionsLeftMargin = 100;
  projectionsRightMargin = 40;

  // 1d distribution overlays
  plotsWidth = 680;
  plotsHeight = 500;
  plotsTopMargin = 50;
  plotsBottomMargin = 80;
  plotsLeftMargin = 100;
  plotsRightMargin = 40;

  // initialize the 'final result' variable (set to false by any failing test)
  finalResult = true;

  // use (near as possible) tdrstyle for plots
  static TStyle style = genStyle();
  style.cd();
}

PlotCompareUtility::~PlotCompareUtility() {
  // close TFiles
  if (refFile != nullptr)
    refFile->Close();
  if (newFile != nullptr)
    newFile->Close();
}

void PlotCompareUtility::dump() {
  cout << "RefFile     = " << refFile->GetName() << endl
       << "RefBasePath = " << refBasePath << endl
       << "RefPrefix   = " << refPrefix << endl
       << "NewFile     = " << newFile->GetName() << endl
       << "NewBasePath = " << newBasePath << endl
       << "NewPrefix   = " << newPrefix << endl
       << "NumHistos   = " << histos.size() << endl;
}

bool PlotCompareUtility::compare(HistoData *HD) {
  bool retval = false;
  switch (HD->getType()) {
    case Plot1D:
      retval = compare<Plot1D>(HD);
      break;
    case Plot2D:
      retval = compare<Plot2D>(HD);
      break;
  }

  finalResult &= retval;
  return retval;
}

void PlotCompareUtility::makePlots(HistoData *HD) {
  switch (HD->getType()) {
    case Plot1D:
      makePlots<Plot1D>(HD);
      break;
    case Plot2D:
      makePlots<Plot2D>(HD);
      break;
  }
}

void PlotCompareUtility::makeHTML(HistoData *HD) {
  switch (HD->getType()) {
    case Plot1D:
      makeHTML<Plot1D>(HD);
      break;
    case Plot2D:
      makeHTML<Plot2D>(HD);
      break;
  }
}

HistoData *PlotCompareUtility::addHistoData(string NewName, string RefName, int Type) {
  // location of this HistoData within the PCU
  static int bin = 0;
  bin++;

  // location of histograms in files
  string newPath = newBasePath + "/" + newPrefix + "/" + NewName;
  string refPath = refBasePath + "/" + refPrefix + "/" + RefName;
  cout << "NewPath     = " << newPath << endl;
  cout << "RefPath     = " << refPath << endl;
  // store the HistoData information
  HistoData hd(NewName, Type, bin, newPath, newFile, refPath, refFile);
  histos.push_back(hd);
  // histos.insert(histos.begin(),hd);
  return &(*histos.rbegin());
}

HistoData *PlotCompareUtility::addProjectionXData(
    HistoData *Parent, std::string Name, int Type, int Bin, TH1 *NewHisto, TH1 *RefHisto) {
  // store the HistoData/projection information
  HistoData hd(Name, Type, Bin, NewHisto, RefHisto);
  projectionsX[Parent].push_back(hd);
  return &(*projectionsX[Parent].rbegin());
}

HistoData *PlotCompareUtility::addProjectionYData(
    HistoData *Parent, std::string Name, int Type, int Bin, TH1 *NewHisto, TH1 *RefHisto) {
  // store the HistoData/projection information
  HistoData hd(Name, Type, Bin, NewHisto, RefHisto);
  projectionsY[Parent].push_back(hd);
  return &(*projectionsY[Parent].rbegin());
}

bool PlotCompareUtility::isValid() const {
  string newPath = newBasePath + "/" + newPrefix;
  string refPath = refBasePath + "/" + refPrefix;
  // check the files and that the paths are valid
  bool refValid = (refFile != nullptr) && (refFile->Get(refPath.c_str()) != nullptr);
  bool newValid = (newFile != nullptr) && (newFile->Get(newPath.c_str()) != nullptr);

  return refValid && newValid;
}

void PlotCompareUtility::centerRebin(TH1 *H1, TH1 *H2) {
  // determine x axis range and binning requirements
  float h1RMS = H1->GetRMS();
  float h2RMS = H2->GetRMS();
  float rms = TMath::Max(h1RMS, h2RMS);
  float h1Mean = H1->GetMean();
  float h2Mean = H2->GetMean();
  float mean = 0.5 * (h1Mean + h2Mean);
  float nBins = H2->GetNbinsX();
  float minX = H2->GetXaxis()->GetXmin();
  float maxX = H2->GetXaxis()->GetXmax();
  float dX = maxX - minX;
  float dNdX = 1;
  float newMinX = 0;
  float newMaxX = 1;

  if (rms > 0) {
    dNdX = 100. / (10 * rms);
    newMinX = mean - 5 * rms;
    newMaxX = mean + 5 * rms;
  }

  // center the histograms onto a common range
  H1->GetXaxis()->SetRangeUser(newMinX, newMaxX);
  H2->GetXaxis()->SetRangeUser(newMinX, newMaxX);

  // rebin the histograms if appropriate
  if (dX * dNdX > 0) {
    int rebinning = (int)(float(nBins) / (dX * dNdX));
    H1->Rebin(rebinning);
    H2->Rebin(rebinning);
  }

  // determine y axis range
  float maxY1 = H1->GetMaximum();
  float maxY2 = H2->GetMaximum();
  float newMinY = 1e-1;
  float newMaxY = maxY1 > maxY2 ? maxY1 : maxY2;
  newMaxY *= 2;

  // ensure that the peaks will be drawn when combined
  H1->GetYaxis()->SetRangeUser(newMinY, newMaxY);
  H2->GetYaxis()->SetRangeUser(newMinY, newMaxY);
}

void PlotCompareUtility::renormalize(TH1 *H1, TH1 *H2) {
  // normalize H2 to H1 (typically this would be hnew to href)
  H2->SetNormFactor(H1->GetEntries());
}

double PlotCompareUtility::getThreshold() const {
  double threshold;

  // the largest non-zero threshold
  if (ksThreshold > 0 && chi2Threshold > 0)
    threshold = ksThreshold < chi2Threshold ? ksThreshold : chi2Threshold;
  else
    threshold = ksThreshold > 0 ? ksThreshold : chi2Threshold;

  return threshold;
}

void PlotCompareUtility::makeSummary(string Name) {
  // produce plot and html
  makeSummaryPlot(Name);
  makeSummaryHTML(Name);
}

void PlotCompareUtility::makeDefaultPlots() {
  // make a default plot for when there i nothing to display
  TCanvas noDataCanvas("noDataCanvas", "noDataCanvas", plotsWidth, plotsHeight);
  noDataCanvas.SetFrameFillColor(10);
  noDataCanvas.Draw();
  TText noData(0.5, 0.5, "No Data");
  noData.Draw();
  noDataCanvas.Print("NoData_Results.gif");
}

void PlotCompareUtility::makeSummaryPlot(string Name) {
  // generate a reasonable height for the summary plot
  int numHistos = histos.size();
  summaryHeight = summaryBottomMargin + summaryTopMargin + int(float(numHistos * summaryBarsThickness) * 1.5);

  // the canvas is rescaled during gif conversion, so add padding to Canvas
  // dimensions
  int summaryCanvasWidth = summaryWidth + 4;
  int summaryCanvasHeight = summaryHeight + 28;

  // create and setup summary canvas
  TCanvas summaryCanvas("summaryCanvas", "summaryCanvas", summaryCanvasWidth, summaryCanvasHeight);
  summaryCanvas.SetFrameFillColor(10);
  summaryCanvas.SetLogx(1);
  summaryCanvas.SetGrid();
  summaryCanvas.SetTopMargin(float(summaryTopMargin) / summaryHeight);
  summaryCanvas.SetLeftMargin(float(summaryLeftMargin) / summaryWidth);
  summaryCanvas.SetRightMargin(float(summaryRightMargin) / summaryWidth);
  summaryCanvas.SetBottomMargin(float(summaryBottomMargin) / summaryHeight);
  summaryCanvas.Draw();

  // create and setup the summary histogram
  TH1F summary("summary", "Compatibility with Reference Histograms", numHistos, 1, numHistos + 1);
  summary.GetXaxis()->SetLabelSize(float(summaryLeftMargin) / (11 * summaryWidth));  // used to be 3*
  summary.GetYaxis()->SetLabelSize(summary.GetXaxis()->GetLabelSize());
  summary.GetYaxis()->SetTitle("Compatibility");
  summary.SetStats(false);
  summary.GetYaxis()->SetRangeUser(getThreshold() / 10, 2);
  summary.Draw("hbar0");

  // loop over hd's and draw result
  vector<HistoData>::iterator hd;
  for (hd = histos.begin(); hd != histos.end(); hd++)
    hd->drawResult(&summary, false, true);

  // draw the pass/fail cutoff line
  TLine passLine(getThreshold(), 1, getThreshold(), numHistos + 1);
  passLine.SetLineColor(kRed);
  passLine.SetLineWidth(2);
  passLine.SetLineStyle(2);
  passLine.Draw("SAME");

  // create the summary image
  string gifName = Name + ".gif";
  summaryCanvas.Print(gifName.c_str());
}

void PlotCompareUtility::makeSummaryHTML(string Name) {
  // create HTML support code
  string gifName = Name + ".gif";
  string html = "index.html";
  ofstream fout(html.c_str());

  // print top portion of document
  fout << "<!DOCTYPE gif PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">" << endl
       << "<html>" << endl
       << endl
       << "  <head>" << endl
       << "    <script type=\"text/javascript\">" << endl
       << "      function tn(target, image) {" << endl
       << "        document.getElementById(\"thumblink\").href = target" << endl
       << "        document.getElementById(\"thumb\").src = image" << endl
       << "      }" << endl
       << "    </script>" << endl
       << "  </head>" << endl
       << endl
       << "  <body>" << endl
       << endl
       << "    <style type=\"text/css\">" << endl
       << "      #thumb_d {position: absolute; position: fixed; left: 5px; "
          "top: 80px; text-align: center; }"
       << endl
       << "      #main_d {position: relative; left: 175px; }" << endl
       << "    </style>" << endl
       << endl
       << "    <!-- added for IE compatibility (untested)" << endl
       << "    <script language=\"JScript\">" << endl
       << "      if (document.recalc && document.body.attachEvent) {" << endl
       << "        theDiv.style.setExpression(\"top\", "
          "\"document.body.scrollTop + 150\"); "
       << endl
       << "        document.body.onscroll = function() { document.recalc(true) "
          "}; "
       << endl
       << "      }" << endl
       << "    </script> -->" << endl
       << endl
       << "    <div id=\"main_d\">" << endl
       << "      <img src=\"" << gifName << "\" usemap=\"#" << Name << "\" alt=\"\" style=\"border-style: none;\""
       << " height=" << summaryHeight << " width=" << summaryWidth << " border=0>" << endl
       << "      <map id=\"" << Name << "\" name=\"" << Name << "\">" << endl;

  // loop over HistoData entries
  vector<HistoData>::iterator hd;
  for (hd = histos.begin(); hd != histos.end(); hd++) {
    // determine map coordinates for this bin (2 pixel offset due to borders?)
    int bin = hd->getBin();
    int x1 = summaryLeftMargin;
    int y1 = summaryHeight - summaryBottomMargin - int(float(bin * 1.5 - .25) * summaryBarsThickness) + 2;
    int x2 = summaryWidth - summaryRightMargin;
    int y2 = y1 + summaryBarsThickness;
    string image = hd->getResultImage();
    string target = hd->getResultTarget();

    // add coordinates area to image map
    fout << "        <area shape=\"rect\" alt=\"\" coords=\"" << x1 << "," << y1 << "," << x2 << "," << y2
         << "\" href=\"" << target << "\" onMouseOver=\"tn('" << target << "','" << image << "')\">" << endl;
  }

  // print bottom portion of document
  fout << "        <area shape=\"default\" nohref=\"nohref\" alt=\"\">" << endl
       << "      </map>" << endl
       << "    </div>" << endl
       << endl
       << "    <div id=\"thumb_d\">" << endl
       << "      <a href=\"#\" id=\"thumblink\"><img src=\"NoData_Results.gif\" "
          "id=\"thumb\" width=200 height=150 border=0></a>"
       << endl
       << "      <br><a href=\"log.txt\">Root Output</a>" << endl
       << "      <br><a href=\"err.txt\">Root Warnings</a>" << endl
       << "    </div>" << endl
       << endl
       << "  </body>" << endl
       << endl
       << "</html>" << endl;

  // close the file
  fout.close();
}
