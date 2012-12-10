
#include <TFile.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TH1.h>
#include <TH2.h>
#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TF1.h>
#include <TPaveText.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TMath.h>
#include <TROOT.h>

#include <string>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <math.h>
#include <limits>

TH1* getHistogram(TFile* inputFile, const std::string& dqmDirectory, const std::string& meName)
{  
  if ( !inputFile ) return 0;

  TString histogramName = "DQMData";
  if ( dqmDirectory != "" ) histogramName.Append(Form("/%s", dqmDirectory.data()));
  if ( histogramName.Length() > 0 && !histogramName.EndsWith("/") ) histogramName.Append("/");
  histogramName.Append(meName);

  TH1* histogram = (TH1*)inputFile->Get(histogramName.Data());
  std::cout << "histogramName = " << histogramName.Data() << ": histogram = " << histogram;
  if ( histogram ) std::cout << ", integral = " << histogram->Integral();
  std::cout << std::endl; 

  if ( !histogram->GetSumw2N() ) histogram->Sumw2();

  //if ( histogram->GetDimension() == 1 ) histogram->Rebin(5);

  histogram->Scale(1./histogram->Integral());

  return histogram;
}

double square(double x)
{
  return x*x;
}

TH1* rebinHistogram(const TH1* histogram, unsigned numBinsMin_rebinned, double xMin, double xMax)
{
  TH1* histogram_rebinned = 0;

  if ( histogram ) {
    unsigned numBins = histogram->GetNbinsX();
    unsigned numBins_withinRange = 0;
    for ( unsigned iBin = 1; iBin <= numBins; ++iBin ) {
      double binCenter = histogram->GetBinCenter(iBin);
      if ( binCenter >= xMin && binCenter <= xMax ) ++numBins_withinRange;
    }

    std::cout << "histogram = " << histogram->GetName() << ":" 
              << " numBins(" << xMin << ".." << "xMax) = " << numBins_withinRange << ", integral = " << histogram->Integral() << std::endl;
    
    unsigned numBins_rebinned = numBins_withinRange;

    for ( int combineNumBins = 5; combineNumBins >= 2; --combineNumBins ) {
      if ( numBins_withinRange >= (combineNumBins*numBinsMin_rebinned) && (numBins % combineNumBins) == 0 ) {
        numBins_rebinned /= combineNumBins;
        numBins_withinRange /= combineNumBins;
      }
    }

    std::string histogramName_rebinned = std::string(histogram->GetName()).append("_rebinned");
    histogram_rebinned = new TH1D(histogramName_rebinned.data(), histogram->GetTitle(), numBins_rebinned, xMin, xMax);
    if ( !histogram_rebinned->GetSumw2N() ) histogram_rebinned->Sumw2();

    TAxis* xAxis = histogram_rebinned->GetXaxis();
      
    unsigned iBin = 1;
    for ( unsigned iBin_rebinned = 1; iBin_rebinned <= numBins_rebinned; ++iBin_rebinned ) {
      double binContent_rebinned = 0.;
      double binError2_rebinned = 0.;

      double xMin_rebinnedBin = xAxis->GetBinLowEdge(iBin_rebinned);
      double xMax_rebinnedBin = xAxis->GetBinUpEdge(iBin_rebinned);

      while ( histogram->GetBinCenter(iBin) < xMin_rebinnedBin ) {
	++iBin;
      }

      while ( histogram->GetBinCenter(iBin) >= xMin_rebinnedBin && histogram->GetBinCenter(iBin) < xMax_rebinnedBin ) {
	binContent_rebinned += histogram->GetBinContent(iBin);
	binError2_rebinned += square(histogram->GetBinError(iBin));
	++iBin;
      }

      histogram_rebinned->SetBinContent(iBin_rebinned, binContent_rebinned);
      histogram_rebinned->SetBinError(iBin_rebinned, TMath::Sqrt(binError2_rebinned));
    }

    std::cout << "histogram(rebinned) = " << histogram_rebinned->GetName() << ":" 
              << " numBins = " << histogram_rebinned->GetNbinsX() << ", integral = " << histogram_rebinned->Integral() << std::endl;
  }

  return histogram_rebinned;
}

//-------------------------------------------------------------------------------
void getBinomialBounds(Int_t n, Int_t r, Float_t& rMin, Float_t& rMax)
{
  rMin = 0.;
  rMax = 0.;

  if ( n == 0 ){
    return;
  }
  if ( r < 0 ){
    std::cerr << "Error in <getBinomialBounds>: n = " << n << ", r = " << r << std::endl;
    return;
  }
  
  if ( ((Double_t)r*(n - r)) > (9.*n) ){
    rMin = r - TMath::Sqrt((Double_t)r*(n - r)/((Double_t)n));
    rMax = r + TMath::Sqrt((Double_t)r*(n - r)/((Double_t)n));
    return;
  }

  Double_t binomialCoefficient = 1.;

  Double_t rMinLeft       = 0.;
  Double_t rMinMiddle     = TMath::Max(0.5*r, n - 1.5*r);
  Double_t rMinRight      = n;
  Double_t rMinLeftProb   = 0.;
  Double_t rMinMiddleProb = 0.5;
  Double_t rMinRightProb  = 1.;
  while ( (rMinRight - rMinLeft) > (0.001*n) ){

    rMinMiddleProb = 0;
    for ( Int_t i = r; i <= n; i++ ){
      binomialCoefficient = 1;

      for ( Int_t j = n; j > i; j-- ){
        binomialCoefficient *= j/((Double_t)(j - i));
      }

      rMinMiddleProb += binomialCoefficient*TMath::Power(rMinMiddle/((Double_t)(n)), i)
                       *TMath::Power((n - rMinMiddle)/((Double_t)(n)), n - i);
    }

    if ( rMinMiddleProb > 0.16 ){
      rMinRight     = rMinMiddle;
      rMinRightProb = rMinMiddleProb;
    } else if ( rMinMiddleProb < 0.16 ){
      rMinLeft      = rMinMiddle;
      rMinLeftProb  = rMinMiddleProb;
    } else {
      rMinLeft      = rMinRight     = rMinMiddle;
      rMinLeftProb  = rMinRightProb = rMinMiddleProb;
    }

    rMinMiddle = 0.5*(rMinLeft + rMinRight);

    if ( rMinLeft > r ){
      rMinMiddle = rMinLeft = rMinRight = 0;
    }
  }

  Double_t rMaxLeft       = 0.;
  Double_t rMaxMiddle     = TMath::Min(1.5*r, n - 0.5*r);
  Double_t rMaxRight      = n;
  Double_t rMaxLeftProb   = 1.;
  Double_t rMaxMiddleProb = 0.5;
  Double_t rMaxRightProb  = 0.;
  while ( (rMaxRight - rMaxLeft) > (0.001*n) ){

    rMaxMiddleProb = 0;
    for ( Int_t i = 0; i <= r; i++ ){
      binomialCoefficient = 1;
      
      for ( Int_t j = n; j > (n - i); j-- ){
        binomialCoefficient *= j/((Double_t)(i - (n - j)));
      }

      rMaxMiddleProb += binomialCoefficient*TMath::Power(rMaxMiddle/((Double_t)(n)), i)
                       *TMath::Power((n - rMaxMiddle)/((Double_t)(n)), n - i);
    }

    if ( rMaxMiddleProb > 0.16 ){
      rMaxLeft      = rMaxMiddle;
      rMaxLeftProb  = rMaxMiddleProb;
    } else if ( rMaxMiddleProb < 0.16 ){
      rMaxRight     = rMaxMiddle;
      rMaxRightProb = rMaxMiddleProb;
    } else {
      rMaxLeft      = rMaxRight     = rMaxMiddle;
      rMaxLeftProb  = rMaxRightProb = rMaxMiddleProb;
    }

    rMaxMiddle = 0.5*(rMaxLeft + rMaxRight);

    if ( rMaxRight < r ){
      rMaxMiddle = rMaxLeft = rMaxRight = n;
    }
  }

  rMin = rMinMiddle;
  rMax = rMaxMiddle;
}

TGraphAsymmErrors* getEfficiency(const TH1* histogram_numerator, const TH1* histogram_denominator)
{
  Int_t error = 0;
  if ( !(histogram_numerator->GetNbinsX()           == histogram_denominator->GetNbinsX())           ) error = 1;
  if ( !(histogram_numerator->GetXaxis()->GetXmin() == histogram_denominator->GetXaxis()->GetXmin()) ) error = 1;
  if ( !(histogram_numerator->GetXaxis()->GetXmax() == histogram_denominator->GetXaxis()->GetXmax()) ) error = 1;
  
  if ( error ){
    std::cerr << "Error in <getEfficiency>: Dimensionality of histograms does not match !!" << std::endl;
    return 0;
  }
  
  TAxis* xAxis = histogram_numerator->GetXaxis();

  Int_t nBins = xAxis->GetNbins();
  TArrayF x(nBins);
  TArrayF dxUp(nBins);
  TArrayF dxDown(nBins);
  TArrayF y(nBins);
  TArrayF dyUp(nBins);
  TArrayF dyDown(nBins);

  for ( Int_t ibin = 1; ibin <= nBins; ibin++ ){
    Int_t nObs = TMath::Nint(histogram_denominator->GetBinContent(ibin));
    Int_t rObs = TMath::Nint(histogram_numerator->GetBinContent(ibin));

    Float_t xCenter = histogram_denominator->GetBinCenter(ibin);
    Float_t xWidth  = histogram_denominator->GetBinWidth(ibin);

    x[ibin - 1]      = xCenter;
    dxUp[ibin - 1]   = 0.5*xWidth;
    dxDown[ibin - 1] = 0.5*xWidth;
    
    if ( nObs > 0 ){
      Float_t rMin = 0.;
      Float_t rMax = 0.;
      
      getBinomialBounds(nObs, rObs, rMin, rMax);

      y[ibin - 1]      = rObs/((Float_t)nObs);
      dyUp[ibin - 1]   = (rMax - rObs)/((Float_t)nObs);
      dyDown[ibin - 1] = (rObs - rMin)/((Float_t)nObs);
    } else{
      y[ibin - 1]      = 0.;
      dyUp[ibin - 1]   = 0.;
      dyDown[ibin - 1] = 0.;
    }
  }
  
  TString name  = TString(histogram_numerator->GetName()).Append("Graph");
  TString title = histogram_numerator->GetTitle();

  TGraphAsymmErrors* graph = 
    new TGraphAsymmErrors(nBins, x.GetArray(), y.GetArray(), 
			  dxDown.GetArray(), dxUp.GetArray(), dyDown.GetArray(), dyUp.GetArray());

  graph->SetName(name);
  graph->SetTitle(title);

  return graph;
}

TGraphAsymmErrors* makeGraph_data_div_mc(const TGraph* graph_data, const TGraph* graph_mc)
{
  TGraphAsymmErrors* graph_data_div_mc = new TGraphAsymmErrors(graph_data->GetN());
  
  for ( int iPoint = 0; iPoint < graph_data->GetN(); ++iPoint ) {
    double x_data, y_data;
    graph_data->GetPoint(iPoint, x_data, y_data);
    double yErrUp_data = graph_data->GetErrorYhigh(iPoint);
    double yErrDown_data = graph_data->GetErrorYlow(iPoint);
    
    double x_mc, y_mc;
    graph_mc->GetPoint(iPoint, x_mc, y_mc);
    double yErrUp_mc = graph_mc->GetErrorYhigh(iPoint);
    double yErrDown_mc = graph_mc->GetErrorYlow(iPoint);
    
    //assert(x_data == x_mc);
    
    if ( !(y_mc > 0.) ) continue;

    double yDiv = (y_data - y_mc)/y_mc;
    double yDivErrUp = 0.;
    if ( y_data > 0. ) yDivErrUp += square(yErrUp_data/y_data);
    if ( y_mc   > 0. ) yDivErrUp += square(yErrDown_mc/y_mc);
    yDivErrUp *= square(y_data/y_mc);
    yDivErrUp = TMath::Sqrt(yDivErrUp);
    double yDivErrDown = 0.;
    if ( y_data > 0. ) yDivErrDown += square(yErrDown_data/y_data);
    if ( y_mc   > 0. ) yDivErrDown += square(yErrUp_mc/y_mc);
    yDivErrDown *= square(y_data/y_mc);
    yDivErrDown = TMath::Sqrt(yDivErrDown);
    
    //std::cout << "x = " << x_data << ": y = " << yDiv << " + " << yDivErrUp << " - " << yDivErrDown << std::endl;
    
    graph_data_div_mc->SetPoint(iPoint, x_data, yDiv);
    graph_data_div_mc->SetPointError(iPoint, 0., 0., yDivErrDown, yDivErrUp);
  }

  return graph_data_div_mc;
}

void showEfficiency(const TString& title, double canvasSizeX, double canvasSizeY,
		    const TH1* histogram_simDYtoTauTau_numerator, const TH1* histogram_simDYtoTauTau_denominator,
		    const TH1* histogram_simDYtoMuMu_genEmbedding_numerator, const TH1* histogram_simDYtoMuMu_genEmbedding_denominator,
		    const TH1* histogram_simDYtoMuMu_recEmbedding_numerator, const TH1* histogram_simDYtoMuMu_recEmbedding_denominator,
		    double xMin, double xMax, unsigned numBinsMin_rebinned, const std::string& xAxisTitle, double xAxisOffset,
                    double yMin, double yMax, double yAxisOffset,
		    const std::string& outputFileName)
{
  TCanvas* canvas = new TCanvas("canvas", "canvas", canvasSizeX, canvasSizeY);
  canvas->SetFillColor(10);
  canvas->SetBorderSize(2);
  canvas->SetLeftMargin(0.12);
  canvas->SetBottomMargin(0.12);

  TPad* topPad = new TPad("topPad", "topPad", 0.00, 0.35, 1.00, 1.00);
  topPad->SetFillColor(10);
  topPad->SetTopMargin(0.04);
  topPad->SetLeftMargin(0.15);
  topPad->SetBottomMargin(0.03);
  topPad->SetRightMargin(0.05);
  topPad->SetGridx();
  topPad->SetGridy();

  canvas->cd();
  topPad->Draw();
  topPad->cd();

  TH1* histogram_simDYtoTauTau_numerator_rebinned = rebinHistogram(histogram_simDYtoTauTau_numerator, numBinsMin_rebinned, xMin, xMax);
  TH1* histogram_simDYtoTauTau_denominator_rebinned = rebinHistogram(histogram_simDYtoTauTau_denominator, numBinsMin_rebinned, xMin, xMax);
  TGraphAsymmErrors* graph_simDYtoTauTau = getEfficiency(histogram_simDYtoTauTau_numerator_rebinned, histogram_simDYtoTauTau_denominator_rebinned);

  TH1* histogram_simDYtoMuMu_genEmbedding_numerator_rebinned = rebinHistogram(histogram_simDYtoMuMu_genEmbedding_numerator, numBinsMin_rebinned, xMin, xMax);
  TH1* histogram_simDYtoMuMu_genEmbedding_denominator_rebinned = rebinHistogram(histogram_simDYtoMuMu_genEmbedding_denominator, numBinsMin_rebinned, xMin, xMax);
  TGraphAsymmErrors* graph_simDYtoMuMu_genEmbedding = getEfficiency(histogram_simDYtoMuMu_genEmbedding_numerator_rebinned, histogram_simDYtoMuMu_genEmbedding_denominator_rebinned);

  TH1* histogram_simDYtoMuMu_recEmbedding_numerator_rebinned = rebinHistogram(histogram_simDYtoMuMu_recEmbedding_numerator, numBinsMin_rebinned, xMin, xMax);
  TH1* histogram_simDYtoMuMu_recEmbedding_denominator_rebinned = rebinHistogram(histogram_simDYtoMuMu_recEmbedding_denominator, numBinsMin_rebinned, xMin, xMax);
  TGraphAsymmErrors* graph_simDYtoMuMu_recEmbedding = getEfficiency(histogram_simDYtoMuMu_recEmbedding_numerator_rebinned, histogram_simDYtoMuMu_recEmbedding_denominator_rebinned);
  
  TH1* dummyHistogram_top = new TH1D("dummyHistogram_top", "dummyHistogram_top", 10, xMin, xMax);
  dummyHistogram_top->SetTitle("");
  dummyHistogram_top->SetStats(false);
  dummyHistogram_top->SetMaximum(yMax);
  dummyHistogram_top->SetMinimum(yMin);
  
  TAxis* xAxis_top = dummyHistogram_top->GetXaxis();
  xAxis_top->SetTitle(xAxisTitle.data());
  xAxis_top->SetTitleOffset(xAxisOffset);
  xAxis_top->SetLabelColor(10);
  xAxis_top->SetTitleColor(10);

  TAxis* yAxis_top = dummyHistogram_top->GetYaxis();
  yAxis_top->SetTitle("#varepsilon");
  yAxis_top->SetTitleOffset(yAxisOffset);

  TLegend* legend = new TLegend(0.61, 0.16, 0.89, 0.47, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);

  dummyHistogram_top->Draw();

  if ( graph_simDYtoTauTau ) {
    graph_simDYtoTauTau->SetLineColor(1);
    graph_simDYtoTauTau->SetMarkerColor(1);
    graph_simDYtoTauTau->SetMarkerStyle(20);
    graph_simDYtoTauTau->Draw("p");

    legend->AddEntry(graph_simDYtoTauTau, "gen. Z/#gamma^{*} #rightarrow #tau #tau", "p");    
  }

  if ( graph_simDYtoMuMu_genEmbedding ) {
    graph_simDYtoMuMu_genEmbedding->SetLineColor(2);
    graph_simDYtoMuMu_genEmbedding->SetMarkerColor(2);
    graph_simDYtoMuMu_genEmbedding->SetMarkerStyle(21);
    graph_simDYtoMuMu_genEmbedding->Draw("p");
    
    legend->AddEntry(graph_simDYtoMuMu_genEmbedding, "Z/#gamma^{*} #rightarrow #mu^{+} #mu^{-}, gen. Embedding", "p");
  }

  if ( graph_simDYtoMuMu_recEmbedding ) {
    graph_simDYtoMuMu_recEmbedding->SetLineColor(4);
    graph_simDYtoMuMu_recEmbedding->SetMarkerColor(4);
    graph_simDYtoMuMu_recEmbedding->SetMarkerStyle(34);
    graph_simDYtoMuMu_recEmbedding->Draw("p");
    
    legend->AddEntry(graph_simDYtoMuMu_recEmbedding, "Z/#gamma^{*} #rightarrow #mu^{+} #mu^{-}, rec. Embedding", "p");
  }

  legend->Draw();

  TPaveText* label = 0;
  if ( title.Length() > 0 ) {
    label = new TPaveText(0.175, 0.89, 0.48, 0.94, "NDC");
    label->AddText(title.Data());
    label->SetTextAlign(13);
    label->SetTextSize(0.045);
    label->SetFillStyle(0);
    label->SetBorderSize(0);
    label->Draw();
  }
  
  TPad* bottomPad = new TPad("bottomPad", "bottomPad", 0.00, 0.00, 1.00, 0.35);
  bottomPad->SetFillColor(10);
  bottomPad->SetTopMargin(0.02);
  bottomPad->SetLeftMargin(0.15);
  bottomPad->SetBottomMargin(0.24);
  bottomPad->SetRightMargin(0.05);
  bottomPad->SetGridx();
  bottomPad->SetGridy();

  canvas->cd();
  bottomPad->Draw();
  bottomPad->cd();

  TH1* dummyHistogram_bottom = new TH1D("dummyHistogram_bottom", "dummyHistogram_bottom", 10, xMin, xMax);
  
  dummyHistogram_bottom->SetMinimum(-1.0);
  dummyHistogram_bottom->SetMaximum(+1.0);

  TAxis* xAxis_bottom = dummyHistogram_bottom->GetXaxis();
  xAxis_bottom->SetTitle(xAxisTitle.data());
  xAxis_bottom->SetTitleOffset(1.20);
  xAxis_bottom->SetLabelColor(1);
  xAxis_bottom->SetTitleColor(1);
  xAxis_bottom->SetTitleSize(0.08);
  xAxis_bottom->SetLabelOffset(0.02);
  xAxis_bottom->SetLabelSize(0.08);
  xAxis_bottom->SetTickLength(0.055);

  TAxis* yAxis_bottom = dummyHistogram_bottom->GetYaxis();
  yAxis_bottom->SetTitle("#frac{Embedding - Z/#gamma^{*} #rightarrow #tau #tau}{Z/#gamma^{*} #rightarrow #tau #tau}");
  yAxis_bottom->SetTitleOffset(0.85);
  yAxis_bottom->SetNdivisions(505);
  yAxis_bottom->CenterTitle();
  yAxis_bottom->SetTitleSize(0.08);
  yAxis_bottom->SetLabelSize(0.08);
  yAxis_bottom->SetTickLength(0.04);

  dummyHistogram_bottom->SetTitle("");
  dummyHistogram_bottom->SetStats(false);
  dummyHistogram_bottom->Draw();
 
  TGraphAsymmErrors* graph_simDYtoMuMu_genEmbedding_div_simDYtoTauTau = 0;
  if ( graph_simDYtoTauTau && graph_simDYtoMuMu_genEmbedding ) {
    graph_simDYtoMuMu_genEmbedding_div_simDYtoTauTau = makeGraph_data_div_mc(graph_simDYtoMuMu_genEmbedding, graph_simDYtoTauTau);
    graph_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->SetLineColor(graph_simDYtoMuMu_genEmbedding->GetLineColor());
    graph_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->SetMarkerColor(graph_simDYtoMuMu_genEmbedding->GetMarkerColor());
    graph_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->SetMarkerStyle(graph_simDYtoMuMu_genEmbedding->GetMarkerStyle());
    graph_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->Draw("p");
  }

  TGraphAsymmErrors* graph_simDYtoMuMu_recEmbedding_div_simDYtoTauTau = 0;
  if ( graph_simDYtoTauTau && graph_simDYtoMuMu_recEmbedding ) {
    graph_simDYtoMuMu_recEmbedding_div_simDYtoTauTau = makeGraph_data_div_mc(graph_simDYtoMuMu_recEmbedding, graph_simDYtoTauTau);
    graph_simDYtoMuMu_recEmbedding_div_simDYtoTauTau->SetLineColor(graph_simDYtoMuMu_recEmbedding->GetLineColor());
    graph_simDYtoMuMu_recEmbedding_div_simDYtoTauTau->SetMarkerColor(graph_simDYtoMuMu_recEmbedding->GetMarkerColor());
    graph_simDYtoMuMu_recEmbedding_div_simDYtoTauTau->SetMarkerStyle(graph_simDYtoMuMu_recEmbedding->GetMarkerStyle());
    graph_simDYtoMuMu_recEmbedding_div_simDYtoTauTau->Draw("p");
  }

  canvas->Update();
  std::string outputFileName_plot = "plots/";
  size_t idx = outputFileName.find_last_of('.');
  outputFileName_plot.append(std::string(outputFileName, 0, idx));
  if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
  canvas->Print(std::string(outputFileName_plot).append(".png").data());
  canvas->Print(std::string(outputFileName_plot).append(".pdf").data());
  
  delete legend;
  delete label;
  delete dummyHistogram_top;
  delete topPad;
  delete dummyHistogram_bottom;
  delete graph_simDYtoMuMu_genEmbedding_div_simDYtoTauTau;
  delete graph_simDYtoMuMu_recEmbedding_div_simDYtoTauTau;
  delete bottomPad;
  delete canvas;
}
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
TH1* compRatioHistogram(const std::string& ratioHistogramName, const TH1* numerator, const TH1* denominator)
{
  assert(numerator->GetDimension() == denominator->GetDimension());
  assert(numerator->GetNbinsX() == denominator->GetNbinsX());

  TH1* histogramRatio = (TH1*)numerator->Clone(ratioHistogramName.data());
  histogramRatio->Divide(denominator);

  Int_t nBins = histogramRatio->GetNbinsX();
  for ( Int_t ibin = 1; ibin <= nBins; ibin++ ){
    double binContent = histogramRatio->GetBinContent(ibin);
    histogramRatio->SetBinContent(ibin, binContent - 1.);
  }

  histogramRatio->SetLineColor(numerator->GetLineColor());
  histogramRatio->SetLineWidth(numerator->GetLineWidth());
  histogramRatio->SetMarkerColor(numerator->GetMarkerColor());
  histogramRatio->SetMarkerStyle(numerator->GetMarkerStyle());

  return histogramRatio;
}

void showDistribution(double canvasSizeX, double canvasSizeY,
		      TH1* histogram_simDYtoTauTau,
		      TH1* histogram_simDYtoMuMu_genEmbedding,
		      TH1* histogram_simDYtoMuMu_recEmbedding,
		      double xMin, double xMax, unsigned numBinsMin_rebinned, const std::string& xAxisTitle, double xAxisOffset,
		      bool useLogScale, double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset,
		      const std::string& outputFileName)
{
  TCanvas* canvas = new TCanvas("canvas", "canvas", canvasSizeX, canvasSizeY);
  canvas->SetFillColor(10);
  canvas->SetBorderSize(2);
  canvas->SetLeftMargin(0.12);
  canvas->SetBottomMargin(0.12);

  TPad* topPad = new TPad("topPad", "topPad", 0.00, 0.35, 1.00, 1.00);
  topPad->SetFillColor(10);
  topPad->SetTopMargin(0.04);
  topPad->SetLeftMargin(0.15);
  topPad->SetBottomMargin(0.03);
  topPad->SetRightMargin(0.05);
  topPad->SetLogy(useLogScale);

  TPad* bottomPad = new TPad("bottomPad", "bottomPad", 0.00, 0.00, 1.00, 0.35);
  bottomPad->SetFillColor(10);
  bottomPad->SetTopMargin(0.02);
  bottomPad->SetLeftMargin(0.15);
  bottomPad->SetBottomMargin(0.24);
  bottomPad->SetRightMargin(0.05);
  bottomPad->SetLogy(false);

  canvas->cd();
  topPad->Draw();
  topPad->cd();

  TH1* histogram_simDYtoTauTau_rebinned = rebinHistogram(histogram_simDYtoTauTau, numBinsMin_rebinned, xMin, xMax);
  TH1* histogram_simDYtoMuMu_genEmbedding_rebinned = rebinHistogram(histogram_simDYtoMuMu_genEmbedding, numBinsMin_rebinned, xMin, xMax);
  TH1* histogram_simDYtoMuMu_recEmbedding_rebinned = rebinHistogram(histogram_simDYtoMuMu_recEmbedding, numBinsMin_rebinned, xMin, xMax);
  
  histogram_simDYtoTauTau_rebinned->SetTitle("");
  histogram_simDYtoTauTau_rebinned->SetStats(false);
  histogram_simDYtoTauTau_rebinned->SetMinimum(yMin);
  histogram_simDYtoTauTau_rebinned->SetMaximum(yMax);
  histogram_simDYtoTauTau_rebinned->SetLineColor(1);
  histogram_simDYtoTauTau_rebinned->SetLineWidth(2);
  histogram_simDYtoTauTau_rebinned->SetMarkerColor(1);
  histogram_simDYtoTauTau_rebinned->SetMarkerStyle(20);
  histogram_simDYtoTauTau_rebinned->Draw("e1p");

  TAxis* xAxis_top = histogram_simDYtoTauTau_rebinned->GetXaxis();
  xAxis_top->SetTitle(xAxisTitle.data());
  xAxis_top->SetTitleOffset(xAxisOffset);
  xAxis_top->SetLabelColor(10);
  xAxis_top->SetTitleColor(10);

  TAxis* yAxis_top = histogram_simDYtoTauTau_rebinned->GetYaxis();
  yAxis_top->SetTitle(yAxisTitle.data());
  yAxis_top->SetTitleOffset(yAxisOffset);

  histogram_simDYtoMuMu_genEmbedding_rebinned->SetLineColor(2);
  histogram_simDYtoMuMu_genEmbedding_rebinned->SetLineWidth(2);
  histogram_simDYtoMuMu_genEmbedding_rebinned->SetMarkerColor(2);
  histogram_simDYtoMuMu_genEmbedding_rebinned->SetMarkerStyle(21);
  histogram_simDYtoMuMu_genEmbedding_rebinned->Draw("e1psame");

  histogram_simDYtoMuMu_recEmbedding_rebinned->SetLineColor(4);
  histogram_simDYtoMuMu_recEmbedding_rebinned->SetLineWidth(2);
  histogram_simDYtoMuMu_recEmbedding_rebinned->SetMarkerColor(4);
  histogram_simDYtoMuMu_recEmbedding_rebinned->SetMarkerStyle(34);
  histogram_simDYtoMuMu_recEmbedding_rebinned->Draw("e1psame");

  TLegend* legend = new TLegend(0.50, 0.74, 0.94, 0.94, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  legend->AddEntry(histogram_simDYtoTauTau_rebinned, "gen. Z/#gamma^{*} #rightarrow #tau #tau", "l");
  legend->AddEntry(histogram_simDYtoMuMu_genEmbedding_rebinned, "Z/#gamma^{*} #rightarrow #mu^{+} #mu^{-}, gen. Embedding", "l");
  legend->AddEntry(histogram_simDYtoMuMu_recEmbedding_rebinned, "Z/#gamma^{*} #rightarrow #mu^{+} #mu^{-}, rec. Embedding", "l");
  legend->Draw();

  canvas->cd();
  bottomPad->Draw();
  bottomPad->cd();

  std::string histogramName_simDYtoMuMu_genEmbedding_div_simDYtoTauTau = std::string(histogram_simDYtoMuMu_genEmbedding->GetName()).append("_div_").append(histogram_simDYtoTauTau->GetName());
  TH1* histogram_simDYtoMuMu_genEmbedding_div_simDYtoTauTau = compRatioHistogram(histogramName_simDYtoMuMu_genEmbedding_div_simDYtoTauTau, histogram_simDYtoMuMu_genEmbedding_rebinned, histogram_simDYtoTauTau_rebinned);
  histogram_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->SetTitle("");
  histogram_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->SetStats(false);
  histogram_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->SetMinimum(-1.);
  histogram_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->SetMaximum(+1.);

  TAxis* xAxis_bottom = histogram_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->GetXaxis();
  xAxis_bottom->SetTitle(xAxis_top->GetTitle());
  xAxis_bottom->SetLabelColor(1);
  xAxis_bottom->SetTitleColor(1);
  xAxis_bottom->SetTitleOffset(1.20);
  xAxis_bottom->SetTitleSize(0.08);
  xAxis_bottom->SetLabelOffset(0.02);
  xAxis_bottom->SetLabelSize(0.08);
  xAxis_bottom->SetTickLength(0.055);
  
  TAxis* yAxis_bottom = histogram_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->GetYaxis();
  yAxis_bottom->SetTitle("#frac{Embedding - Z/#gamma^{*} #rightarrow #tau #tau}{Z/#gamma^{*} #rightarrow #tau #tau}");
  yAxis_bottom->SetTitleOffset(0.70);
  yAxis_bottom->SetNdivisions(505);
  yAxis_bottom->CenterTitle();
  yAxis_bottom->SetTitleSize(0.08);
  yAxis_bottom->SetLabelSize(0.08);
  yAxis_bottom->SetTickLength(0.04);  
  
  histogram_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->Draw("axis");

  std::string histogramName_simDYtoMuMu_recEmbedding_div_simDYtoTauTau = std::string(histogram_simDYtoMuMu_recEmbedding->GetName()).append("_div_").append(histogram_simDYtoTauTau->GetName());
  TH1* histogram_simDYtoMuMu_recEmbedding_div_simDYtoTauTau = compRatioHistogram(histogramName_simDYtoMuMu_recEmbedding_div_simDYtoTauTau, histogram_simDYtoMuMu_recEmbedding_rebinned, histogram_simDYtoTauTau_rebinned);
  histogram_simDYtoMuMu_recEmbedding_div_simDYtoTauTau->SetTitle("");
  histogram_simDYtoMuMu_recEmbedding_div_simDYtoTauTau->SetStats(false);
  histogram_simDYtoMuMu_recEmbedding_div_simDYtoTauTau->SetMinimum(-1.);
  histogram_simDYtoMuMu_recEmbedding_div_simDYtoTauTau->SetMaximum(+1.);
  
  histogram_simDYtoMuMu_genEmbedding_div_simDYtoTauTau->Draw("e1psame");
  
  histogram_simDYtoMuMu_recEmbedding_div_simDYtoTauTau->Draw("e1psame");
  
  canvas->Update();
  size_t idx = outputFileName.find_last_of('.');
  std::string outputFileName_plot = std::string(outputFileName, 0, idx);
  if ( useLogScale ) outputFileName_plot.append("_log");
  else outputFileName_plot.append("_linear");
  if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
  canvas->Print(std::string(outputFileName_plot).append(".png").data());
  //canvas->Print(std::string(outputFileName_plot).append(".pdf").data());
  
  delete legend;
  delete histogram_simDYtoMuMu_genEmbedding_div_simDYtoTauTau;
  delete histogram_simDYtoMuMu_recEmbedding_div_simDYtoTauTau;
  delete canvas;  
}
//-------------------------------------------------------------------------------

struct plotEntryType
{
  plotEntryType(const std::string& name, const std::string& title, const std::string& meName, 
		double xMin, double xMax, unsigned numBinsMin_rebinned, const std::string& xAxisTitle, double xAxisOffset,
		double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset, bool useLogScale)
    : name_(name),
      title_(title),
      meName_(meName),
      xMin_(xMin),
      xMax_(xMax),
      numBinsMin_rebinned_(numBinsMin_rebinned),
      xAxisTitle_(xAxisTitle),
      xAxisOffset_(xAxisOffset),
      yMin_(yMin),
      yMax_(yMax),
      yAxisTitle_(yAxisTitle),
      yAxisOffset_(yAxisOffset),
      useLogScale_(useLogScale)
  {}
  ~plotEntryType() {}
  std::string name_;
  std::string title_;
  std::string meName_;
  double xMin_;
  double xMax_;
  unsigned numBinsMin_rebinned_;
  std::string xAxisTitle_;
  double xAxisOffset_;
  double yMin_;
  double yMax_;
  std::string yAxisTitle_;
  double yAxisOffset_;
  bool useLogScale_;
};

void makeEmbeddingValidationPlots()
{
//--- stop ROOT from keeping references to all histograms
  TH1::AddDirectory(false);

//--- suppress the output canvas 
  gROOT->SetBatch(true);

  std::string inputFilePath = "/data1/veelken/tmp/EmbeddingValidation/";

  std::string inputFileName_simDYtoTauTau            = "validateMCEmbedding_simDYtoTauTau_all_v1_4_8.root";
  std::string inputFileName_simDYtoMuMu_genEmbedding = "validateMCEmbedding_simDYtoMuMu_genEmbedding_all_v1_4_8.root";
  std::string inputFileName_simDYtoMuMu_recEmbedding = "validateMCEmbedding_simDYtoMuMu_recEmbedding_all_v1_4_8.root";

  std::vector<plotEntryType> distributionsToPlot;
  distributionsToPlot.push_back(plotEntryType("numGlobalMuons", "numGlobalMuons", "numGlobalMuons", -0.5, +9.5, 10, "Num. global Muons", 1.2, 0., 1., "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("numStandAloneMuons", "numStandAloneMuons", "numStandAloneMuons", -0.5, +9.5, 10, "Num. stand-alone Muons", 1.2, 0., 1., "a.u.", 1.2, false));

  distributionsToPlot.push_back(plotEntryType("numTracksPtGt5", "numTracksPtGt5", "numTracksPtGt5", -0.5, +49.5, 50, "Num. Tracks of P_{T} > 5 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType("numTracksPtGt10", "numTracksPtGt10", "numTracksPtGt10", -0.5, +34.5, 50, "Num. Tracks of P_{T} > 10 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType("numTracksPtGt20", "numTracksPtGt20", "numTracksPtGt20", -0.5, +19.5, 50, "Num. Tracks of P_{T} > 20 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  
  distributionsToPlot.push_back(plotEntryType("genMuonEta", "genMuonEta", "validationAnalyzer_mutau/goodMuonDistributions/genLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("genMuonPt",  "genMuonPt",  "validationAnalyzer_mutau/goodMuonDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("genMuonPt",  "genMuonPt",  "validationAnalyzer_mutau/goodMuonDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType("recMuonEta", "recMuonEta", "validationAnalyzer_mutau/goodMuonDistributions/recLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("recMuonPt",  "recMuonPt",  "validationAnalyzer_mutau/goodMuonDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("recMuonPt",  "recMuonPt",  "validationAnalyzer_mutau/goodMuonDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));

  distributionsToPlot.push_back(plotEntryType("genIsoMuonEta", "genIsoMuonEta", "validationAnalyzer_mutau/goodIsoMuonDistributions/genLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("genIsoMuonPt",  "genIsoMuonPt",  "validationAnalyzer_mutau/goodIsoMuonDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("genIsoMuonPt",  "genIsoMuonPt",  "validationAnalyzer_mutau/goodIsoMuonDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType("recIsoMuonEta", "recIsoMuonEta", "validationAnalyzer_mutau/goodIsoMuonDistributions/recLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("recIsoMuonPt",  "recIsoMuonPt",  "validationAnalyzer_mutau/goodIsoMuonDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("recIsoMuonPt",  "recIsoMuonPt",  "validationAnalyzer_mutau/goodIsoMuonDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));

  distributionsToPlot.push_back(plotEntryType("genTauEta", "genTauEta", "validationAnalyzer_mutau/selectedTauDistributions/genLeptonEta", -2.5, +2.5, 50, "#eta_{#tau}", 1.2, 0., 0.1, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("genTauPt",  "genTauPt",  "validationAnalyzer_mutau/selectedTauDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#tau}", 1.3, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("genTauPt",  "genTauPt",  "validationAnalyzer_mutau/selectedTauDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#tau}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType("recTauEta", "recTauEta", "validationAnalyzer_mutau/selectedTauDistributions/recLeptonEta", -2.5, +2.5, 50, "#eta_{#tau}", 1.2, 0., 0.1, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("recTauPt",  "recTauPt",  "validationAnalyzer_mutau/selectedTauDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#tau}", 1.3, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("recTauPt",  "recTauPt",  "validationAnalyzer_mutau/selectedTauDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#tau}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  
  distributionsToPlot.push_back(plotEntryType("genMEt", "genMEt", "validationAnalyzer_mutau/type1CorrPFMEtDistributions/genMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("genMEt", "genMEt", "validationAnalyzer_mutau/type1CorrPFMEtDistributions/genMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType("recCaloMEtNoHF", "recCaloMEtNoHF", "validationAnalyzer_mutau/rawCaloMEtNoHFdistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("recCaloMEtNoHF", "recCaloMEtNoHF", "validationAnalyzer_mutau/rawCaloMEtNoHFdistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType("recPFMEt", "recPFMEt", "validationAnalyzer_mutau/rawPFMEtDistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("recPFMEt", "recPFMEt", "validationAnalyzer_mutau/rawPFMEtDistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType("recPFMEtTypeIcorrected", "recPFMEtTypeIcorrected", "validationAnalyzer_mutau/type1CorrPFMEtDistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType("recPFMEtTypeIcorrected", "recPFMEtTypeIcorrected", "validationAnalyzer_mutau/type1CorrPFMEtDistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));

  TFile* inputFile_simDYtoTauTau = new TFile(std::string(inputFilePath).append(inputFileName_simDYtoTauTau).data());
  TFile* inputFile_simDYtoMuMu_genEmbedding = new TFile(std::string(inputFilePath).append(inputFileName_simDYtoMuMu_genEmbedding).data());
  TFile* inputFile_simDYtoMuMu_recEmbedding = new TFile(std::string(inputFilePath).append(inputFileName_simDYtoMuMu_recEmbedding).data());

  for ( std::vector<plotEntryType>::const_iterator plot = distributionsToPlot.begin();
	plot != distributionsToPlot.end(); ++plot ) {
    TH1* histogram_simDYtoTauTau = getHistogram(inputFile_simDYtoTauTau, "", plot->meName_);
    TH1* histogram_simDYtoMuMu_genEmbedding = getHistogram(inputFile_simDYtoMuMu_genEmbedding, "", plot->meName_);
    TH1* histogram_simDYtoMuMu_recEmbedding = getHistogram(inputFile_simDYtoMuMu_recEmbedding, "", plot->meName_);
    std::string outputFileName = Form("plots/makeEmbeddingValidationPlots_%s.pdf", plot->name_.data());
    showDistribution(800, 900,
		     histogram_simDYtoTauTau,
		     histogram_simDYtoMuMu_genEmbedding,
		     histogram_simDYtoMuMu_recEmbedding,
		     plot->xMin_, plot->xMax_, plot->numBinsMin_rebinned_, plot->xAxisTitle_, plot->xAxisOffset_,
		     plot->useLogScale_, plot->yMin_, plot->yMax_, plot->yAxisTitle_, plot->yAxisOffset_,
		     outputFileName);
  }

  delete inputFile_simDYtoTauTau;
  delete inputFile_simDYtoMuMu_genEmbedding;
  delete inputFile_simDYtoMuMu_recEmbedding;
}
