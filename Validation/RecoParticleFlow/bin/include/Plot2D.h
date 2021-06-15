#ifndef PLOT_2D__H
#define PLOT_2D__H

#include "Plot1D.h"
#include "PlotCompareUtility.h"
#include "PlotTypes.h"

#include <TGaxis.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLine.h>
#include <TProfile.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

template <>
inline bool PlotCompareUtility::compare<Plot2D>(HistoData *HD) {
  // get the reference and comparison histograms
  TH2F *href2d = (TH2F *)HD->getRefHisto();
  TH2F *hnew2d = (TH2F *)HD->getNewHisto();

  // do not run comparisons if either histogram is empty/broken
  if (hnew2d == nullptr || href2d == nullptr || hnew2d->GetEntries() <= 1 || href2d->GetEntries() <= 1) {
    // std::cerr << HD->getName() << " error: unable to retrieve histogram (or
    // no entries)\n";
    HD->setIsEmpty(true);
    return false;
  }

  // prepare an overall result
  bool projectionsPassed = true;

  // loop over axes (projections on one or both may be requested)
  for (int axis = axisX; axis <= axisY; ++axis) {
    // for X: verify projections requested and proper Y binning of href2d and
    // hnew2d
    if (axis == axisX && !HD->getDoProjectionsX())
      continue;
    if (axis == axisX && href2d->GetNbinsY() != hnew2d->GetNbinsY()) {
      std::cerr << HD->getName() << " error: incorrect number of bins for X projection tests\n";
      projectionsPassed = false;
      continue;
    }

    // for Y: verify projections requested and proper X binning of href2d and
    // hnew2d
    if (axis == axisY && !HD->getDoProjectionsY())
      continue;
    if (axis == axisY && href2d->GetNbinsX() != hnew2d->GetNbinsX()) {
      std::cerr << HD->getName() << " error: incorrect number of bins for Y projection tests\n";
      projectionsPassed = false;
      continue;
    }

    // setup the rebinning variables
    int nBins = (axis == axisX) ? href2d->GetNbinsY() : href2d->GetNbinsX();
    int nProjections = (axis == axisX) ? HD->getMaxProjectionsX() : HD->getMaxProjectionsY();
    int nGroups = (int)ceil(float(nBins) / nProjections);
    bool rebinned = false;

    // for X projections: if required rebin a clone of href2d and hnew2d
    if (axis == axisX && HD->getDoAllow2DRebinningX() && nGroups > 1) {
      href2d = (TH2F *)(((TH2F *)(href2d->Clone()))->RebinY(nGroups));
      hnew2d = (TH2F *)(((TH2F *)(hnew2d->Clone()))->RebinY(nGroups));
      nBins = href2d->GetNbinsY();
      rebinned = true;
    }

    // for Y projections: if required rebin a clone of href2d and hnew2d
    if (axis == axisY && HD->getDoAllow2DRebinningY() && nGroups > 1) {
      href2d = (TH2F *)(((TH2F *)(href2d->Clone()))->RebinX(nGroups));
      hnew2d = (TH2F *)(((TH2F *)(hnew2d->Clone()))->RebinX(nGroups));
      nBins = href2d->GetNbinsX();
      rebinned = true;
    }

    // loop over bins in histograms (go backwords to keep in order)
    // for (int bin = nBins; bin >= 1; --bin) {
    for (int bin = 1; bin <= nBins; ++bin) {
      std::cout << "bin " << bin << " of " << nBins << std::endl;
      // create some unique identifiers for the histogram names
      TString projName = HD->getName() + (axis == axisX ? "_px" : "_py");
      projName += bin;
      TString newProjName = "new_";
      newProjName += projName;
      TString refProjName = "ref_";
      refProjName += projName;

      // get the 1d projections for this bin out of the histogram
      TH1D *hnew = (axis == axisX) ? hnew2d->ProjectionX(newProjName.Data(), bin, bin)
                                   : hnew2d->ProjectionY(newProjName.Data(), bin, bin);
      TH1D *href = (axis == axisX) ? href2d->ProjectionX(refProjName.Data(), bin, bin)
                                   : href2d->ProjectionY(refProjName.Data(), bin, bin);

      // set histogram axis labels
      hnew->GetXaxis()->SetTitle((axis == axisX ? hnew2d->GetXaxis()->GetTitle() : hnew2d->GetYaxis()->GetTitle()));
      href->GetXaxis()->SetTitle((axis == axisX ? href2d->GetXaxis()->GetTitle() : href2d->GetYaxis()->GetTitle()));

      // allow Root to delete these histograms after display
      hnew->SetBit(kCanDelete);
      href->SetBit(kCanDelete);

      // create a new HistoData based on this projection
      HistoData *proj = (axis == axisX) ? addProjectionXData(HD, projName.Data(), Plot1D, bin, hnew, href)
                                        : addProjectionYData(HD, projName.Data(), Plot1D, bin, hnew, href);

      // ignore empty bins
      // if (hnew->Integral() == 0 || href->Integral() == 0) continue;
      if (hnew->GetEntries() <= 1 || href->GetEntries() <= 1 || hnew->Integral() == 0 || href->Integral() == 0)
        continue;

      // run this new HistoData through compare<Plot1D>
      projectionsPassed &= compare<Plot1D>(proj);

      // get the high and low scores from this comparison
      float lowScore = proj->getLowScore();
      float highScore = proj->getHighScore();
      if (lowScore < HD->getLowScore())
        HD->setLowScore(lowScore);
      if (highScore > HD->getHighScore())
        HD->setHighScore(highScore);
    }

    // if 2d histograms were rebinned, delete the clone and re-get the original
    if (rebinned) {
      delete href2d;
      href2d = (TH2F *)HD->getRefHisto();
      delete hnew2d;
      hnew2d = (TH2F *)HD->getNewHisto();
    }
  }

  // check overall result
  HD->setResult(projectionsPassed);
  HD->setIsEmpty(false);

  // returns true on test passed and false on test failed
  return projectionsPassed;
}

template <>
inline void PlotCompareUtility::makePlots<Plot2D>(HistoData *HD) {
  // do not make any new plot if empty
  if (HD->getIsEmpty()) {
    HD->setResultImage("NoData_Results.gif");
    HD->setResultTarget("NoData_Results.gif");
    return;
  }

  // loop over the projections to make 1D plots
  std::vector<HistoData>::iterator hd;
  for (hd = projectionsX[HD].begin(); hd != projectionsX[HD].end(); hd++)
    makePlots<Plot1D>(&(*hd));
  for (hd = projectionsY[HD].begin(); hd != projectionsY[HD].end(); hd++)
    makePlots<Plot1D>(&(*hd));

  // make projection summaries
  for (int axis = axisX; axis <= axisY; ++axis) {
    // get the list of projections associated with this HistoData
    std::vector<HistoData> *proj = (axis == axisX) ? &projectionsX[HD] : &projectionsY[HD];
    if (proj == nullptr || proj->empty())
      continue;

    // get the 2d histograms
    TH2F *hnew2d = (TH2F *)HD->getNewHisto();
    TH2F *href2d = (TH2F *)HD->getRefHisto();

    // generate a reasonable width for the projections summary
    int numHistos = proj->size();
    int bodyWidth = int(float(numHistos * projectionsBarsThickness) * 1.5);
    projectionsWidth = projectionsLeftMargin + projectionsRightMargin + bodyWidth;

    // the canvas is rescaled during gif conversion, so add padding to Canvas
    // dimensions
    int projectionsCanvasWidth = projectionsWidth + 4;
    int projectionsCanvasHeight = projectionsHeight + 28;

    // create and setup projections canvas
    TCanvas projectionsCanvas(
        "projectionsCanvas", "projectionsCanvas", projectionsCanvasWidth, projectionsCanvasHeight);
    projectionsCanvas.SetFrameFillColor(10);
    projectionsCanvas.SetLogy(1);
    projectionsCanvas.SetTopMargin(float(projectionsTopMargin) / projectionsHeight);
    projectionsCanvas.SetLeftMargin(float(projectionsLeftMargin) / projectionsWidth);
    projectionsCanvas.SetRightMargin(float(projectionsRightMargin) / projectionsWidth);
    projectionsCanvas.SetBottomMargin(float(projectionsBottomMargin) / projectionsHeight);
    projectionsCanvas.Draw();

    // create and setup the summary histogram
    TH1F projectionsSummary(
        "projectionsSummary", "Compatibility with Reference Histograms", numHistos, 1, numHistos + 1);
    projectionsSummary.GetYaxis()->SetRangeUser(getThreshold() / 10, 2);
    projectionsSummary.SetStats(false);

    // display histogram (take axis from original histogram)
    projectionsSummary.Draw("AH");

    // draw X axis
    float xMin = hnew2d->GetXaxis()->GetXmin();
    float xMax = hnew2d->GetXaxis()->GetXmax();
    int ticksNDiv = numHistos * 20 + bodyWidth / 50;  // formerly *20
    TGaxis *xAxis = new TGaxis(1, 0, numHistos + 1, 0, xMin, xMax, ticksNDiv, "");
    if (axis == axisX)
      xAxis->SetTitle(hnew2d->GetYaxis()->GetTitle());
    if (axis == axisY)
      xAxis->SetTitle(hnew2d->GetXaxis()->GetTitle());
    xAxis->Draw();

    // draw Y axis
    float yMin = getThreshold() / 10;
    float yMax = 2;
    TGaxis *yAxis = new TGaxis(1, yMin, 1, yMax, yMin, yMax, 510, "G");
    yAxis->SetTitle("Compatibility");
    yAxis->Draw();

    // loop over projections and draw result
    std::vector<HistoData>::iterator pd;
    for (pd = proj->begin(); pd != proj->end(); pd++)
      pd->drawResult(&projectionsSummary, true, false);

    // draw the pass/fail cutoff line
    TLine passLine(1, getThreshold(), numHistos + 1, getThreshold());
    passLine.SetLineColor(kRed);
    passLine.SetLineWidth(2);
    passLine.SetLineStyle(2);
    passLine.Draw("SAME");

    // create the summary image
    std::string gifName = HD->getName() + (axis == axisX ? "_Results_px.gif" : "_Results_py.gif");
    projectionsCanvas.Print(gifName.c_str());

    // make overall projection plot of the original 2d histogram
    std::string projName = HD->getName() + (axis == axisX ? "_py" : "_px");
    std::string newBinsProj = projName + "_new";
    std::string refBinsProj = projName + "_ref";
    TH1D *href = (axis == axisX) ? href2d->ProjectionY(refBinsProj.c_str()) : href2d->ProjectionX(refBinsProj.c_str());
    TH1D *hnew = (axis == axisX) ? hnew2d->ProjectionY(newBinsProj.c_str()) : hnew2d->ProjectionX(newBinsProj.c_str());

    // allow Root to delete these histograms after display
    href->SetBit(kCanDelete);
    hnew->SetBit(kCanDelete);

    // create a new HistoData based on this projection and plot it
    HistoData allBins(projName, Plot1D, 0, hnew, href);
    allBins.setIsEmpty(false);
    allBins.setShadedFillColor(HD->getShadedFillColor());
    allBins.setShadedFillStyle(HD->getShadedFillStyle());
    allBins.setShadedLineColor(HD->getShadedLineColor());
    makePlots<Plot1D>(&allBins);

    // set the default image (axisY takes priority by default)
    if (HD->getResultImage().empty() || axis == axisY)
      HD->setResultImage(projName + "_Results.gif");

    // set the default target (in case additional HTML code is/was not produced)
    std::string currentTarget = HD->getResultTarget();
    std::string xImgTarget = HD->getName() + "_px_Results.gif";
    if (currentTarget.empty() || (axis == axisY && currentTarget == xImgTarget))
      HD->setResultTarget(projName + "_Results.gif");
  }

  /*
  // make overall profile plot of the original 2d histogram
  for (int axis = axisX; axis <= axisY; ++axis) {

    // make profile plots out of original 2D histograms
    TProfile *pref = (axis == axisX) ? ((TH2F *)HD->getRefHisto())->ProfileY() :
  ((TH2F *)HD->getRefHisto())->ProfileX(); TProfile *pnew = (axis == axisX) ?
  ((TH2F *)HD->getNewHisto())->ProfileY() : ((TH2F
  *)HD->getNewHisto())->ProfileX();

    // renormalize results for display
        renormalize(pref,pnew);

    // do not allow Root to deallocate this memory after drawing (tries to free
  twice?) pref->SetBit(kCanDelete); pnew->SetBit(kCanDelete);

    // set drawing options on the reference histogram
    pref->SetStats(0);
    pref->SetLineColor(kBlack);
    pref->SetMarkerColor(kBlack);

    // set drawing options on the new histogram
    pnew->SetStats(0);
    pnew->SetLineColor(HD->getSolidLineColor());

    // place the test results as the title
    TString title = HD->getName();

    // the canvas is rescaled during gif conversion, so add padding to Canvas
  dimensions int plotsCanvasWidth = plotsWidth + 4; int plotsCanvasHeight =
  plotsHeight + 28;

    // setup canvas for displaying the compared histograms
    TCanvas hCanvas("hCanvas",title.Data(),plotsCanvasWidth,plotsCanvasHeight);
    hCanvas.SetTopMargin(float(plotsTopMargin) / plotsHeight);
    hCanvas.SetLeftMargin(float(plotsLeftMargin) / plotsWidth);
    hCanvas.SetRightMargin(float(plotsRightMargin) / plotsWidth);
    hCanvas.SetBottomMargin(float(plotsBottomMargin) / plotsHeight);
    hCanvas.SetFrameFillColor(10);
    hCanvas.SetGrid();
    //hCanvas.SetLogy(1);
    hCanvas.Draw();

    // draw the profiles
    pref->Draw();
    pnew->Draw("SAME");
    if (HD->getDoDrawErrorBars()) pnew->Draw("E1SAME");

    // draw a legend
    TLegend legend(0.15,0.01,0.3, 0.08);
    legend.AddEntry(pnew,"New","lF");
    legend.AddEntry(pref,"Reference","lF");
    legend.SetFillColor(kNone);
    legend.Draw("SAME");

    // create the plots overlay image
    std::string gifName = HD->getName() + (axis == axisX ? "_pfx.gif" :
  "_pfy.gif"); hCanvas.Print(gifName.c_str());

    // set the default image (axisY takes priority by default)
    if (HD->getResultImage() == "" || axis == axisY)
  HD->setResultImage(gifName);

    // set the default target (in case additional HTML code is/was not produced)
    std::string currentTarget = HD->getResultTarget();
    std::string xImgTarget = HD->getName() + "_pfx.gif";
    if (currentTarget == "" || (axis == axisY && currentTarget == xImgTarget))
  HD->setResultTarget(gifName);

  }
  */
}

template <>
inline void PlotCompareUtility::makeHTML<Plot2D>(HistoData *HD) {
  /* at present, makeHTML<Plot1D> does nothing, so don't waste the CPU cycles
  // loop over projections and produce HTML
  std::vector<HistoData>::iterator hd;
  for (hd = projectionsX[HD].begin(); hd != projectionsX[HD].end(); hd++)
  makePlots<Plot1D>(&(*hd)); for (hd = projectionsY[HD].begin(); hd !=
  projectionsY[HD].end(); hd++) makePlots<Plot1D>(&(*hd));
  */

  // get the HistoData name for later reuse
  std::string Name = HD->getName();

  // loop over the axes to see if projections were produced
  bool pfDone[2] = {false, false};
  for (int axis = axisX; axis <= axisY; axis++) {
    // get the list of projections associated with this HistoData
    std::vector<HistoData> *proj = (axis == axisX) ? &projectionsX[HD] : &projectionsY[HD];
    if (proj == nullptr || proj->empty())
      continue;
    else
      pfDone[axis] = true;

    // setup some names, etc. for insertion into the HTML
    std::string gifNameProjections = Name + (axis == axisX ? "_Results_px.gif" : "_Results_py.gif");
    std::string gifNameAllProj = Name + (axis == axisX ? "_py_Results.gif" : "_px_Results.gif");
    std::string gifNameProfile = Name + (axis == axisX ? "_pfx.gif" : "_pfy.gif");
    std::string gifBinPrefix = Name + (axis == axisX ? "_px" : "_py");

    // setup some locations to put thumbnails, etc.
    int offset = 10;
    int thumbWidth = plotsWidth / 4;
    int thumbHeight = plotsHeight / 4;
    int bodyWidth = projectionsWidth - projectionsLeftMargin - projectionsRightMargin;
    int leftThumbPos = offset + projectionsLeftMargin + bodyWidth / 4 - thumbWidth / 2;
    int rightThumbPos = leftThumbPos + bodyWidth / 2;
    int thumbsLoc = projectionsTopMargin + thumbHeight / 2;

    // create the profile's HTML document
    std::string htmlNameProfile = Name + (axis == axisX ? "_Results_px.html" : "_Results_py.html");
    std::ofstream fout(htmlNameProfile.c_str());

    // print top portion of document
    fout << "<!DOCTYPE gif PUBLIC '-//W3C//DTD HTML 4.01 Transitional//EN'>" << std::endl
         << "<html>" << std::endl
         << "  <head>" << std::endl
         << "    <title>Compatibility of Projections for " << HD->getRefHisto()->GetTitle() << "</title>" << std::endl
         << "    <script type='text/javascript'>" << std::endl
         << std::endl
         << "      function tn(target,image,class) {" << std::endl
         << "        clear()" << std::endl
         << "        "
            "document.getElementById('thumb_div').setAttribute('class',class)"
         << std::endl
         << "        "
            "document.getElementById('thumb_div').setAttribute('className',"
            "class)"
         << std::endl
         << "        document.getElementById('thumb_link').href = target" << std::endl
         << "        document.getElementById('thumb_img').src = image" << std::endl
         << "        document.getElementById('thumb_img').width = '" << thumbWidth << "'" << std::endl
         << "        document.getElementById('thumb_img').height = '" << thumbHeight << "'" << std::endl
         << "        document.getElementById('thumb_img').border = '1'" << std::endl
         << "      }" << std::endl
         << std::endl
         << "      function clear() {" << std::endl
         << "        document.getElementById('thumb_link').href = '#'" << std::endl
         << "        document.getElementById('thumb_img').src = ''" << std::endl
         << "        document.getElementById('thumb_img').width = '0'" << std::endl
         << "        document.getElementById('thumb_img').height = '0'" << std::endl
         << "        document.getElementById('thumb_img').border = '0'" << std::endl
         << "      }" << std::endl
         << std::endl
         << "    </script>" << std::endl
         << "  </head>" << std::endl
         << "  <body onClick=\"window.location.href='index.html'\">"
         << std::endl
         //         << "<a href='index.html'>"
         << "    <style type='text/css'>" << std::endl
         << "      #thumb_div {}" << std::endl
         << "      div.thumb_left {position: absolute; left: " << leftThumbPos << "px; top: " << thumbsLoc << "px;}"
         << std::endl
         << "      div.thumb_right {position: absolute; left: " << rightThumbPos << "px; top: " << thumbsLoc << "px;}"
         << std::endl
         << "      #main_d {position: absolute; left: " << offset << "px;}" << std::endl
         << "      a:link {color: #000000}" << std::endl
         << "      a:visited {color: #000000}" << std::endl
         << "      a:hover {color: #000000}" << std::endl
         << "      a:active {color: #000000}" << std::endl
         << "    </style>" << std::endl
         << "    <div id='main_d'>"
         << std::endl
         // << " <p>" <<   HD->getRefHisto()->GetTitle() << "</p>"  //include
         // the Title of the Plot as a title of the page
         << "      <img src='" << gifNameProjections << "' usemap='#results' alt=''"
         << " height=" << projectionsHeight << " width=" << projectionsWidth << " border=0>" << std::endl
         << "      <map id='#results' name='results' onMouseOut=\"clear()\">" << std::endl;

    // loop over projections
    std::vector<HistoData>::iterator pd;
    for (pd = proj->begin(); pd != proj->end(); pd++) {
      // determine map coordinates for this bin (1 pixel offset due to borders?)
      int bin = pd->getBin();
      int x1 = projectionsLeftMargin + int(float(bin * 1.5 - 1.25) * projectionsBarsThickness);
      int x2 = x1 + projectionsBarsThickness;
      int y1 = projectionsTopMargin + 1;
      int y2 = projectionsHeight - projectionsBottomMargin;
      std::string image = pd->getResultImage();
      std::string target = pd->getResultTarget();

      // add coordinates area to image map
      std::string tnClass = (bin - 1 >= float(proj->size()) / 2 ? "thumb_left" : "thumb_right");
      fout << "        <area shape='rect' alt='' coords='" << x1 << "," << y1 << "," << x2 << "," << y2 << "'"
           << " href='" << target << "' onMouseOver=\"tn('" << target << "','" << image << "','" << tnClass << "')\" "
           << "onMouseDown=\"window.location.href='" << target << "'\">" << std::endl;
    }

    fout << "        <area shape='default' nohref='nohref' "
            "onMouseDown='window.location.reload()' alt=''>"
         << std::endl
         << "      </map>" << std::endl
         << "      <br><img src=\"" << gifNameAllProj << "\">" << std::endl
         << "    </div>" << std::endl
         << "    <div id='thumb_div'><a href='#' id='thumb_link'><img src='' "
            "id='thumb_img' width=0 height=0 border=0></a></div>"
         << std::endl
         //         << " </a>"
         << "  </body>" << std::endl
         << "</html>" << std::endl;

    // close the file
    HD->setResultTarget(htmlNameProfile);
    fout.close();
  }

  // if both profile dimensions were filled, we need an additional HTML document
  if (pfDone[axisX] && pfDone[axisY]) {
    // create HTML support code for this HistoData
    std::string html = Name + "_Results_Profiles.html";
    std::ofstream fout(html.c_str());

    // make a simple frames portal to show both profile
    fout << "<html>" << std::endl
         << "  <frameset rows=\"50%,50%\">" << std::endl
         << "    <frame src=\"" << Name << "_Results_py.html\">" << std::endl
         << "    <frame src=\"" << Name << "_Results_px.html\">" << std::endl
         << "    <noframes><body>" << std::endl
         << "      unable to display frames -- click to select desired page" << std::endl
         << "      <br><a href =\"" << Name << "_Results_py.html\">Y Projections</a>" << std::endl
         << "      <br><a href =\"" << Name << "_Results_px.html\">X Projections</a>" << std::endl
         << "    </body></noframes>" << std::endl
         << "  </frameset>" << std::endl
         << "</html>" << std::endl;

    // close the file
    HD->setResultTarget(html);
    fout.close();
  }
}

#endif  // PLOT_2D__H
