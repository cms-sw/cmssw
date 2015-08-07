
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
  if ( !histogram ) {
    std::cerr << "Failed to find histogram = " << histogramName.Data() << " in inputFile = " << inputFile->GetName() << " !!" << std::endl;
    assert(0);
  }

  std::cout << "histogram = " << histogramName.Data() << ": integral = " << histogram->Integral() << std::endl; 

  if ( !histogram->GetSumw2N() ) histogram->Sumw2();

  //if ( histogram->GetDimension() == 1 ) histogram->Rebin(5);

  //histogram->Scale(1./histogram->Integral());

  return histogram;
}

double square(double x)
{
  return x*x;
}

TH1* rebinHistogram(const TH1* histogram, unsigned numBinsMin_rebinned, double xMin, double xMax, bool normalize)
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

    if ( normalize ) {
      if ( !histogram_rebinned->GetSumw2N() ) histogram_rebinned->Sumw2();
      histogram_rebinned->Scale(1./histogram_rebinned->Integral());
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
  
  const TAxis* xAxis = histogram_numerator->GetXaxis();

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

TGraphAsymmErrors* makeGraph_div_ref(const TGraph* graph, const TGraph* graph_ref)
{
  TGraphAsymmErrors* graph_div_ref = new TGraphAsymmErrors(graph->GetN());
  
  for ( int iPoint = 0; iPoint < graph->GetN(); ++iPoint ) {
    double x, y;
    graph->GetPoint(iPoint, x, y);
    double yErrUp = graph->GetErrorYhigh(iPoint);
    double yErrDown = graph->GetErrorYlow(iPoint);
    
    double x_ref, y_ref;
    graph_ref->GetPoint(iPoint, x_ref, y_ref);
    double yErrUp_ref = graph_ref->GetErrorYhigh(iPoint);
    double yErrDown_ref = graph_ref->GetErrorYlow(iPoint);
        
    if ( !(y_ref > 0.) ) continue;

    double yDiv = (y - y_ref)/y_ref;
    double yDivErrUp = 0.;
    if ( y     > 0. ) yDivErrUp += square(yErrUp/y);
    if ( y_ref > 0. ) yDivErrUp += square(yErrDown_ref/y_ref);
    yDivErrUp *= square(y/y_ref);
    yDivErrUp = TMath::Sqrt(yDivErrUp);
    double yDivErrDown = 0.;
    if ( y     > 0. ) yDivErrDown += square(yErrDown/y);
    if ( y_ref > 0. ) yDivErrDown += square(yErrUp_ref/y_ref);
    yDivErrDown *= square(y/y_ref);
    yDivErrDown = TMath::Sqrt(yDivErrDown);
    
    //std::cout << "x = " << x << ": y = " << yDiv << " + " << yDivErrUp << " - " << yDivErrDown << std::endl;
    
    graph_div_ref->SetPoint(iPoint, x, yDiv);
    graph_div_ref->SetPointError(iPoint, 0., 0., yDivErrDown, yDivErrUp);
  }

  graph_div_ref->SetLineColor(graph->GetLineColor());
  graph_div_ref->SetMarkerColor(graph->GetMarkerColor());
  graph_div_ref->SetMarkerStyle(graph->GetMarkerStyle());
  graph_div_ref->SetMarkerSize(graph->GetMarkerSize());

  return graph_div_ref;
}

void showEfficiency(const TString& title, double canvasSizeX, double canvasSizeY,
		    const TH1* histogram_ref_numerator, const TH1* histogram_ref_denominator, const std::string& legendEntry_ref,
		    const TH1* histogram2_numerator, const TH1* histogram2_denominator, const std::string& legendEntry2,
		    const TH1* histogram3_numerator, const TH1* histogram3_denominator, const std::string& legendEntry3,
		    const TH1* histogram4_numerator, const TH1* histogram4_denominator, const std::string& legendEntry4,
		    const TH1* histogram5_numerator, const TH1* histogram5_denominator, const std::string& legendEntry5,
	   	    const TH1* histogram6_numerator, const TH1* histogram6_denominator, const std::string& legendEntry6,
		    double xMin, double xMax, unsigned numBinsMin_rebinned, const std::string& xAxisTitle, double xAxisOffset,
                    double yMin, double yMax, double yAxisOffset,
		    double legendX0, double legendY0, 
		    const std::string& outputFileName)
{
  TCanvas* canvas = new TCanvas("canvas", "canvas", canvasSizeX, canvasSizeY);
  canvas->SetFillColor(10);
  canvas->SetBorderSize(2);
  canvas->SetLeftMargin(0.135);
  canvas->SetBottomMargin(0.135);

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

  dummyHistogram_top->Draw();

  int colors[6] = { kBlack, kGreen - 6, kBlue - 7,  kMagenta - 7, kCyan - 6, kRed - 6 };
  int markerStyles[6] = { 24, 25, 20, 21, 22, 23 };
  int markerSizes[6] = { 1, 1, 1, 1, 1, 1 };

  TLegend* legend = new TLegend(legendX0, legendY0, legendX0 + 0.44, legendY0 + 0.20, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  
  TH1* histogram_ref_numerator_rebinned = rebinHistogram(histogram_ref_numerator, numBinsMin_rebinned, xMin, xMax, false);
  TH1* histogram_ref_denominator_rebinned = rebinHistogram(histogram_ref_denominator, numBinsMin_rebinned, xMin, xMax, false);
  TGraphAsymmErrors* graph_ref = getEfficiency(histogram_ref_numerator_rebinned, histogram_ref_denominator_rebinned);
  graph_ref->SetLineColor(colors[0]);
  graph_ref->SetMarkerColor(colors[0]);
  graph_ref->SetMarkerStyle(markerStyles[0]);
  graph_ref->SetMarkerSize(markerSizes[0]);
  graph_ref->Draw("p");
  legend->AddEntry(graph_ref, legendEntry_ref.data(), "p");    

  TGraphAsymmErrors* graph2 = 0;
  if ( histogram2_numerator && histogram2_denominator ) {
    TH1* histogram2_numerator_rebinned = rebinHistogram(histogram2_numerator, numBinsMin_rebinned, xMin, xMax, false);
    TH1* histogram2_denominator_rebinned = rebinHistogram(histogram2_denominator, numBinsMin_rebinned, xMin, xMax, false);
    graph2 = getEfficiency(histogram2_numerator_rebinned, histogram2_denominator_rebinned);
    graph2->SetLineColor(colors[1]);
    graph2->SetMarkerColor(colors[1]);
    graph2->SetMarkerStyle(markerStyles[1]);
    graph2->SetMarkerSize(markerSizes[1]);
    graph2->Draw("p");
    legend->AddEntry(graph2, legendEntry2.data(), "p");
  }

  TGraphAsymmErrors* graph3 = 0;
  if ( histogram3_numerator && histogram3_denominator ) {
    TH1* histogram3_numerator_rebinned = rebinHistogram(histogram3_numerator, numBinsMin_rebinned, xMin, xMax, false);
    TH1* histogram3_denominator_rebinned = rebinHistogram(histogram3_denominator, numBinsMin_rebinned, xMin, xMax, false);
    graph3 = getEfficiency(histogram3_numerator_rebinned, histogram3_denominator_rebinned);
    graph3->SetLineColor(colors[2]);
    graph3->SetMarkerColor(colors[2]);
    graph3->SetMarkerStyle(markerStyles[2]);
    graph3->SetMarkerSize(markerSizes[2]);
    graph3->Draw("p");
    legend->AddEntry(graph3, legendEntry3.data(), "p");
  }
  
  TGraphAsymmErrors* graph4 = 0;
  if ( histogram4_numerator && histogram4_denominator ) {
    TH1* histogram4_numerator_rebinned = rebinHistogram(histogram4_numerator, numBinsMin_rebinned, xMin, xMax, false);
    TH1* histogram4_denominator_rebinned = rebinHistogram(histogram4_denominator, numBinsMin_rebinned, xMin, xMax, false);
    graph4 = getEfficiency(histogram4_numerator_rebinned, histogram4_denominator_rebinned);
    graph4->SetLineColor(colors[3]);
    graph4->SetMarkerColor(colors[3]);
    graph4->SetMarkerStyle(markerStyles[3]);
    graph4->SetMarkerSize(markerSizes[3]);
    graph4->Draw("p");
    legend->AddEntry(graph4, legendEntry4.data(), "p");
  }

  TGraphAsymmErrors* graph5 = 0;
  if ( histogram5_numerator && histogram5_denominator ) {
    TH1* histogram5_numerator_rebinned = rebinHistogram(histogram5_numerator, numBinsMin_rebinned, xMin, xMax, false);
    TH1* histogram5_denominator_rebinned = rebinHistogram(histogram5_denominator, numBinsMin_rebinned, xMin, xMax, false);
    graph5 = getEfficiency(histogram5_numerator_rebinned, histogram5_denominator_rebinned);
    graph5->SetLineColor(colors[4]);
    graph5->SetMarkerColor(colors[4]);
    graph5->SetMarkerStyle(markerStyles[4]);
    graph5->SetMarkerSize(markerSizes[4]);
    graph5->Draw("p");
    legend->AddEntry(graph5, legendEntry5.data(), "p");
  }
 
  TGraphAsymmErrors* graph6 = 0;
  if ( histogram6_numerator && histogram6_denominator ) {
    TH1* histogram6_numerator_rebinned = rebinHistogram(histogram6_numerator, numBinsMin_rebinned, xMin, xMax, false);
    TH1* histogram6_denominator_rebinned = rebinHistogram(histogram6_denominator, numBinsMin_rebinned, xMin, xMax, false);
    graph6 = getEfficiency(histogram6_numerator_rebinned, histogram6_denominator_rebinned);
    graph6->SetLineColor(colors[5]);
    graph6->SetMarkerColor(colors[5]);
    graph6->SetMarkerStyle(markerStyles[5]);
    graph6->SetMarkerSize(markerSizes[5]);
    graph6->Draw("p");
    legend->AddEntry(graph6, legendEntry6.data(), "p");
  }

  legend->Draw();

  TPaveText* label = 0;
  if ( title.Length() > 0 ) {
    label = new TPaveText(0.145, 0.975, 0.48, 1.02, "NDC");
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
  
  dummyHistogram_bottom->SetMinimum(-0.50);
  dummyHistogram_bottom->SetMaximum(+0.50);

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
 
  TGraphAsymmErrors* graph2_div_ref = 0;
  if ( graph2 ) {
    graph2_div_ref = makeGraph_div_ref(graph2, graph_ref);
    graph2_div_ref->Draw("p");
  }

  TGraphAsymmErrors* graph3_div_ref = 0;
  if ( graph3 ) {
    graph3_div_ref = makeGraph_div_ref(graph3, graph_ref);
    graph3_div_ref->Draw("p");
  }

  TGraphAsymmErrors* graph4_div_ref = 0;
  if ( graph4 ) {
    graph4_div_ref = makeGraph_div_ref(graph4, graph_ref);
    graph4_div_ref->Draw("p");
  }

  TGraphAsymmErrors* graph5_div_ref = 0;
  if ( graph5 ) {
    graph5_div_ref = makeGraph_div_ref(graph5, graph_ref);
    graph5_div_ref->Draw("p");
  }

  TGraphAsymmErrors* graph6_div_ref = 0;
  if ( graph6 ) {
    graph6_div_ref = makeGraph_div_ref(graph6, graph_ref);
    graph6_div_ref->Draw("p");
  }

  canvas->Update();
  size_t idx = outputFileName.find_last_of('.');
  std::string outputFileName_plot = std::string(outputFileName, 0, idx);
  if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
  canvas->Print(std::string(outputFileName_plot).append(".png").data());
  canvas->Print(std::string(outputFileName_plot).append(".pdf").data());
  
  delete legend;
  delete label;
  delete dummyHistogram_top;
  delete topPad;
  delete dummyHistogram_bottom;
  delete graph2_div_ref;
  delete graph3_div_ref;
  delete graph4_div_ref;
  delete graph5_div_ref;
  delete graph6_div_ref;  
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

  int nBins = histogramRatio->GetNbinsX();
  for ( int iBin = 1; iBin <= nBins; ++iBin ){
    double binContent = histogramRatio->GetBinContent(iBin);
    histogramRatio->SetBinContent(iBin, binContent - 1.);
  }

  histogramRatio->SetLineColor(numerator->GetLineColor());
  histogramRatio->SetLineWidth(numerator->GetLineWidth());
  histogramRatio->SetMarkerColor(numerator->GetMarkerColor());
  histogramRatio->SetMarkerStyle(numerator->GetMarkerStyle());
  histogramRatio->SetMarkerSize(numerator->GetMarkerSize());

  return histogramRatio;
}

void showDistribution(double canvasSizeX, double canvasSizeY,
		      TH1* histogram_ref, const std::string& legendEntry_ref,
		      TH1* histogram2, const std::string& legendEntry2,
		      TH1* histogram3, const std::string& legendEntry3,
		      TH1* histogram4, const std::string& legendEntry4,
		      TH1* histogram5, const std::string& legendEntry5,
		      TH1* histogram6, const std::string& legendEntry6,
		      double xMin, double xMax, unsigned numBinsMin_rebinned, const std::string& xAxisTitle, double xAxisOffset,
		      bool useLogScale, double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset,
		      double legendX0, double legendY0, 
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

  int colors[6] = { kBlack, kGreen - 6, kBlue - 7,  kMagenta - 7, kCyan - 6, kRed - 6 };
  int markerStyles[6] = { 24, 25, 20, 21, 22, 23 };
  int markerSizes[6] = { 1, 1, 1, 1, 1, 1 };

  TLegend* legend = new TLegend(legendX0, legendY0, legendX0 + 0.44, legendY0 + 0.20, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);

  TH1* histogram_ref_rebinned = rebinHistogram(histogram_ref, numBinsMin_rebinned, xMin, xMax, false);
  histogram_ref_rebinned->SetTitle("");
  histogram_ref_rebinned->SetStats(false);
  histogram_ref_rebinned->SetMinimum(yMin);
  histogram_ref_rebinned->SetMaximum(yMax);
  histogram_ref_rebinned->SetLineColor(colors[0]);
  histogram_ref_rebinned->SetLineWidth(2);
  histogram_ref_rebinned->SetMarkerColor(colors[0]);
  histogram_ref_rebinned->SetMarkerStyle(markerStyles[0]);
  histogram_ref_rebinned->SetMarkerSize(markerSizes[0]);
  histogram_ref_rebinned->Draw("e1p");
  legend->AddEntry(histogram_ref_rebinned, legendEntry_ref.data(), "p");

  TAxis* xAxis_top = histogram_ref_rebinned->GetXaxis();
  xAxis_top->SetTitle(xAxisTitle.data());
  xAxis_top->SetTitleOffset(xAxisOffset);
  xAxis_top->SetLabelColor(10);
  xAxis_top->SetTitleColor(10);

  TAxis* yAxis_top = histogram_ref_rebinned->GetYaxis();
  yAxis_top->SetTitle(yAxisTitle.data());
  yAxis_top->SetTitleOffset(yAxisOffset);

  TH1* histogram2_rebinned = 0;
  if ( histogram2 ) {
    histogram2_rebinned = rebinHistogram(histogram2, numBinsMin_rebinned, xMin, xMax, false);
    histogram2_rebinned->SetLineColor(colors[1]);
    histogram2_rebinned->SetLineWidth(2);
    histogram2_rebinned->SetMarkerColor(colors[1]);
    histogram2_rebinned->SetMarkerStyle(markerStyles[1]);
    histogram2_rebinned->SetMarkerSize(markerSizes[1]);
    histogram2_rebinned->Draw("e1psame");
    legend->AddEntry(histogram2_rebinned, legendEntry2.data(), "p");
  }

  TH1* histogram3_rebinned = 0;
  if ( histogram3 ) {
    histogram3_rebinned = rebinHistogram(histogram3, numBinsMin_rebinned, xMin, xMax, false);
    histogram3_rebinned->SetLineColor(colors[2]);
    histogram3_rebinned->SetLineWidth(2);
    histogram3_rebinned->SetMarkerColor(colors[2]);
    histogram3_rebinned->SetMarkerStyle(markerStyles[2]);
    histogram3_rebinned->SetMarkerSize(markerSizes[2]);
    histogram3_rebinned->Draw("e1psame");
    legend->AddEntry(histogram3_rebinned, legendEntry3.data(), "p");
  }

  TH1* histogram4_rebinned = 0;
  if ( histogram4 ) {
    histogram4_rebinned = rebinHistogram(histogram4, numBinsMin_rebinned, xMin, xMax, false);
    histogram4_rebinned->SetLineColor(colors[3]);
    histogram4_rebinned->SetLineWidth(2);
    histogram4_rebinned->SetMarkerColor(colors[3]);
    histogram4_rebinned->SetMarkerStyle(markerStyles[3]);
    histogram4_rebinned->SetMarkerSize(markerSizes[3]);
    histogram4_rebinned->Draw("e1psame");
    legend->AddEntry(histogram4_rebinned, legendEntry4.data(), "p");
  }

  TH1* histogram5_rebinned = 0;
  if ( histogram5 ) {
    histogram5_rebinned = rebinHistogram(histogram5, numBinsMin_rebinned, xMin, xMax, false);
    histogram5_rebinned->SetLineColor(colors[4]);
    histogram5_rebinned->SetLineWidth(2);
    histogram5_rebinned->SetMarkerColor(colors[4]);
    histogram5_rebinned->SetMarkerStyle(markerStyles[4]);
    histogram5_rebinned->SetMarkerSize(markerSizes[4]);
    histogram5_rebinned->Draw("e1psame");
    legend->AddEntry(histogram5_rebinned, legendEntry5.data(), "p");
  }

  TH1* histogram6_rebinned = 0;
  if ( histogram6 ) {
    histogram6_rebinned = rebinHistogram(histogram6, numBinsMin_rebinned, xMin, xMax, false);
    histogram6_rebinned->SetLineColor(colors[5]);
    histogram6_rebinned->SetLineWidth(2);
    histogram6_rebinned->SetMarkerColor(colors[5]);
    histogram6_rebinned->SetMarkerStyle(markerStyles[5]);
    histogram6_rebinned->SetMarkerSize(markerSizes[5]);
    histogram6_rebinned->Draw("e1psame");
    legend->AddEntry(histogram6_rebinned, legendEntry6.data(), "p");
  }

  legend->Draw();

  canvas->cd();
  bottomPad->Draw();
  bottomPad->cd();

  TH1* histogram2_div_ref = 0;
  if ( histogram2 ) {
    std::string histogramName2_div_ref = std::string(histogram2->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram2_div_ref = compRatioHistogram(histogramName2_div_ref, histogram2_rebinned, histogram_ref_rebinned);
    histogram2_div_ref->SetTitle("");
    histogram2_div_ref->SetStats(false);
    histogram2_div_ref->SetMinimum(-0.50);
    histogram2_div_ref->SetMaximum(+0.50);

    TAxis* xAxis_bottom = histogram2_div_ref->GetXaxis();
    xAxis_bottom->SetTitle(xAxis_top->GetTitle());
    xAxis_bottom->SetLabelColor(1);
    xAxis_bottom->SetTitleColor(1);
    xAxis_bottom->SetTitleOffset(1.20);
    xAxis_bottom->SetTitleSize(0.08);
    xAxis_bottom->SetLabelOffset(0.02);
    xAxis_bottom->SetLabelSize(0.08);
    xAxis_bottom->SetTickLength(0.055);
    
    TAxis* yAxis_bottom = histogram2_div_ref->GetYaxis();
    yAxis_bottom->SetTitle("#frac{Embedding - Z/#gamma^{*} #rightarrow #tau #tau}{Z/#gamma^{*} #rightarrow #tau #tau}");
    yAxis_bottom->SetTitleOffset(0.70);
    yAxis_bottom->SetNdivisions(505);
    yAxis_bottom->CenterTitle();
    yAxis_bottom->SetTitleSize(0.08);
    yAxis_bottom->SetLabelSize(0.08);
    yAxis_bottom->SetTickLength(0.04);  
  
    histogram2_div_ref->Draw("e1p");
  }

  TH1* histogram3_div_ref = 0;
  if ( histogram3 ) {
    std::string histogramName3_div_ref = std::string(histogram3->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram3_div_ref = compRatioHistogram(histogramName3_div_ref, histogram3_rebinned, histogram_ref_rebinned);
    histogram3_div_ref->SetTitle("");
    histogram3_div_ref->SetStats(false);
    histogram3_div_ref->SetMinimum(-0.50);
    histogram3_div_ref->SetMaximum(+0.50);

    TAxis* xAxis_bottom = histogram3_div_ref->GetXaxis();
    xAxis_bottom->SetTitle(xAxis_top->GetTitle());
    xAxis_bottom->SetLabelColor(1);
    xAxis_bottom->SetTitleColor(1);
    xAxis_bottom->SetTitleOffset(1.20);
    xAxis_bottom->SetTitleSize(0.08);
    xAxis_bottom->SetLabelOffset(0.02);
    xAxis_bottom->SetLabelSize(0.08);
    xAxis_bottom->SetTickLength(0.055);
    
    TAxis* yAxis_bottom = histogram3_div_ref->GetYaxis();
    yAxis_bottom->SetTitle("#frac{Embedding - Z/#gamma^{*} #rightarrow #tau #tau}{Z/#gamma^{*} #rightarrow #tau #tau}");
    yAxis_bottom->SetTitleOffset(0.70);
    yAxis_bottom->SetNdivisions(505);
    yAxis_bottom->CenterTitle();
    yAxis_bottom->SetTitleSize(0.08);
    yAxis_bottom->SetLabelSize(0.08);
    yAxis_bottom->SetTickLength(0.04);  

    if ( histogram2 ) histogram3_div_ref->Draw("e1psame");
    else histogram3_div_ref->Draw("e1p");
  }

  TH1* histogram4_div_ref = 0;
  if ( histogram4 ) {
    std::string histogramName4_div_ref = std::string(histogram4->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram4_div_ref = compRatioHistogram(histogramName4_div_ref, histogram4_rebinned, histogram_ref_rebinned);
    histogram4_div_ref->Draw("e1psame");
  }

  TH1* histogram5_div_ref = 0;
  if ( histogram5 ) {
    std::string histogramName5_div_ref = std::string(histogram5->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram5_div_ref = compRatioHistogram(histogramName5_div_ref, histogram5_rebinned, histogram_ref_rebinned);
    histogram5_div_ref->Draw("e1psame");
  }

  TH1* histogram6_div_ref = 0;
  if ( histogram6 ) {
    std::string histogramName6_div_ref = std::string(histogram6->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram6_div_ref = compRatioHistogram(histogramName6_div_ref, histogram6_rebinned, histogram_ref_rebinned);
    histogram6_div_ref->Draw("e1psame");
  }
  
  canvas->Update();
  size_t idx = outputFileName.find_last_of('.');
  std::string outputFileName_plot = std::string(outputFileName, 0, idx);
  if ( useLogScale ) outputFileName_plot.append("_log");
  else outputFileName_plot.append("_linear");
  if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
  canvas->Print(std::string(outputFileName_plot).append(".png").data());
  canvas->Print(std::string(outputFileName_plot).append(".pdf").data());
  
  delete legend;
  delete histogram2_div_ref;
  delete histogram3_div_ref;
  delete histogram4_div_ref;
  delete histogram5_div_ref;
  delete histogram6_div_ref;
  delete topPad;
  delete bottomPad;
  delete canvas;  
}
//-------------------------------------------------------------------------------

struct plotEntryTypeBase
{
  plotEntryTypeBase(const std::string& name, const std::string& title, 
		    double xMin, double xMax, unsigned numBinsMin_rebinned, const std::string& xAxisTitle, double xAxisOffset,
		    double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset, bool useLogScale)
    : name_(name),
      title_(title),
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
  ~plotEntryTypeBase() {}
  std::string name_;
  std::string title_;
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

struct plotEntryType_distribution : public plotEntryTypeBase
{
  plotEntryType_distribution(const std::string& name, const std::string& title, const std::string& dqmDirectory, const std::string& meName, 
			     double xMin, double xMax, unsigned numBinsMin_rebinned, const std::string& xAxisTitle, double xAxisOffset,
			     double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset, bool useLogScale)
    : plotEntryTypeBase(name, title, 
			xMin, xMax, numBinsMin_rebinned, xAxisTitle, xAxisOffset, 
			yMin, yMax, yAxisTitle, yAxisOffset, useLogScale)
  {
    TString meName_full = dqmDirectory.data();
    if ( meName_full.Length() > 0 && !meName_full.EndsWith("/") ) meName_full.Append("/");
    meName_full.Append(meName.data());
    meName_ = meName_full.Data();
  }
  ~plotEntryType_distribution() {}
  std::string meName_;  
};

struct plotEntryType_efficiency : public plotEntryTypeBase
{
  plotEntryType_efficiency(const std::string& name, const std::string& title, const std::string& dqmDirectory, const std::string& meName_numerator, const std::string& meName_denominator, 
			   double xMin, double xMax, unsigned numBinsMin_rebinned, const std::string& xAxisTitle, double xAxisOffset,
			   double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset, bool useLogScale)
    : plotEntryTypeBase(name, title, 
			xMin, xMax, numBinsMin_rebinned, xAxisTitle, xAxisOffset, 
			yMin, yMax, yAxisTitle, yAxisOffset, useLogScale)
  {
    TString meName_numerator_full = dqmDirectory.data();
    if ( meName_numerator_full.Length() > 0 && !meName_numerator_full.EndsWith("/") ) meName_numerator_full.Append("/");
    meName_numerator_full.Append(meName_numerator.data());
    meName_numerator_ = meName_numerator_full.Data();
    TString meName_denominator_full = dqmDirectory.data();
    if ( meName_denominator_full.Length() > 0 && !meName_denominator_full.EndsWith("/") ) meName_denominator_full.Append("/");
    meName_denominator_full.Append(meName_denominator.data());
    meName_denominator_ = meName_denominator_full.Data();
  }
  ~plotEntryType_efficiency() {}
  std::string meName_numerator_;
  std::string meName_denominator_;
};

//-------------------------------------------------------------------------------
TGraphAsymmErrors* convertToGraph(const TH1* histogram, const TH1* histogramErrUp = 0, const TH1* histogramErrDown = 0)
{
  if ( histogramErrUp   ) assert(histogram->GetNbinsX() == histogramErrUp->GetNbinsX());
  if ( histogramErrDown ) assert(histogram->GetNbinsX() == histogramErrDown->GetNbinsX());
  int nPoints = histogram->GetNbinsX();

  TGraphAsymmErrors* graph = new TGraphAsymmErrors(nPoints);

  int nBins = histogram->GetNbinsX();
  for ( int iBin = 1; iBin <= nBins; ++iBin ){
    double x = histogram->GetBinCenter(iBin);
    double y = histogram->GetBinContent(iBin);
    double xErrUp = 0.5*histogram->GetBinWidth(iBin);
    double xErrDown = xErrUp;
    double yErr2Up = square(histogram->GetBinError(iBin));
    if ( histogramErrUp   && histogramErrUp->GetBinContent(iBin)   > y ) yErr2Up = square(histogramErrUp->GetBinContent(iBin)   - y);
    if ( histogramErrDown && histogramErrDown->GetBinContent(iBin) > y ) yErr2Up = square(histogramErrDown->GetBinContent(iBin) - y);
    double yErrUp = TMath::Sqrt(yErr2Up);
    double yErr2Down = square(histogram->GetBinError(iBin));
    if ( histogramErrUp   && histogramErrUp->GetBinContent(iBin)   < y ) yErr2Down = square(y - histogramErrUp->GetBinContent(iBin));
    if ( histogramErrDown && histogramErrDown->GetBinContent(iBin) < y ) yErr2Down = square(y - histogramErrDown->GetBinContent(iBin));
    double yErrDown = TMath::Sqrt(yErr2Down);

    int iPoint = iBin - 1;
    graph->SetPoint(iPoint, x, y);
    graph->SetPointError(iPoint, xErrDown, xErrUp, yErrDown, yErrUp);
  }

  graph->SetLineColor(histogram->GetLineColor());
  graph->SetLineWidth(histogram->GetLineWidth());
  graph->SetMarkerColor(histogram->GetMarkerColor());
  graph->SetMarkerStyle(histogram->GetMarkerStyle());
  graph->SetMarkerSize(histogram->GetMarkerSize());

  return graph;
}

TGraphAsymmErrors* compRatioGraph(const std::string& ratioGraphName, const TGraph* numerator, const TGraph* denominator)
{
  assert(numerator->GetN() == denominator->GetN());
  int nPoints = numerator->GetN();

  TGraphAsymmErrors* graphRatio = new TGraphAsymmErrors(nPoints);
  graphRatio->SetName(ratioGraphName.data());

  for ( int iPoint = 0; iPoint < nPoints; ++iPoint ){
    double x_numerator, y_numerator;
    numerator->GetPoint(iPoint, x_numerator, y_numerator);
    double xErrUp_numerator = 0.;
    double xErrDown_numerator = 0.;
    double yErrUp_numerator = 0.;
    double yErrDown_numerator = 0.;
    if ( dynamic_cast<const TGraphAsymmErrors*>(numerator) ) {
      const TGraphAsymmErrors* numerator_asymmerrors = dynamic_cast<const TGraphAsymmErrors*>(numerator);
      xErrUp_numerator = numerator_asymmerrors->GetErrorXhigh(iPoint);
      xErrDown_numerator = numerator_asymmerrors->GetErrorXlow(iPoint);
      yErrUp_numerator = numerator_asymmerrors->GetErrorYhigh(iPoint);
      yErrDown_numerator = numerator_asymmerrors->GetErrorYlow(iPoint);
    } else if ( dynamic_cast<const TGraphErrors*>(numerator) ) {
      const TGraphErrors* numerator_errors = dynamic_cast<const TGraphErrors*>(numerator);
      xErrUp_numerator = numerator_errors->GetErrorX(iPoint);
      xErrDown_numerator = xErrUp_numerator;
      yErrUp_numerator = numerator_errors->GetErrorY(iPoint);
      yErrDown_numerator = yErrUp_numerator;
    }

    double x_denominator, y_denominator;
    denominator->GetPoint(iPoint, x_denominator, y_denominator);
    assert(x_denominator == x_numerator);
    double xErrUp_denominator = 0.;
    double xErrDown_denominator = 0.;
    double yErrUp_denominator = 0.;
    double yErrDown_denominator = 0.;
    if ( dynamic_cast<const TGraphAsymmErrors*>(denominator) ) {
      const TGraphAsymmErrors* denominator_asymmerrors = dynamic_cast<const TGraphAsymmErrors*>(denominator);
      xErrUp_denominator = denominator_asymmerrors->GetErrorXhigh(iPoint);
      xErrDown_denominator = denominator_asymmerrors->GetErrorXlow(iPoint);
      yErrUp_denominator = denominator_asymmerrors->GetErrorYhigh(iPoint);
      yErrDown_denominator = denominator_asymmerrors->GetErrorYlow(iPoint);
    } else if ( dynamic_cast<const TGraphErrors*>(denominator) ) {
      const TGraphErrors* denominator_errors = dynamic_cast<const TGraphErrors*>(denominator);
      xErrUp_denominator = denominator_errors->GetErrorX(iPoint);
      xErrDown_denominator = xErrUp_denominator;
      yErrUp_denominator = denominator_errors->GetErrorY(iPoint);
      yErrDown_denominator = yErrUp_denominator;
    }

    double x_ratio = x_numerator;
    double y_ratio = ( y_denominator > 0. ) ? (y_numerator/y_denominator) : 0.;
    double xErrUp_ratio = TMath::Max(xErrUp_numerator, xErrUp_denominator);
    double xErrDown_ratio = TMath::Max(xErrDown_numerator, xErrDown_denominator);
    double yErr2Up_ratio = 0.;
    if ( y_numerator   ) yErr2Up_ratio += square(yErrUp_numerator/y_numerator);
    if ( y_denominator ) yErr2Up_ratio += square(yErrDown_denominator/y_numerator);
    double yErrUp_ratio = TMath::Sqrt(yErr2Up_ratio)*y_ratio;
    double yErr2Down_ratio = 0.;
    if ( y_numerator   ) yErr2Down_ratio += square(yErrDown_numerator/y_numerator);
    if ( y_denominator ) yErr2Down_ratio += square(yErrUp_denominator/y_numerator);
    double yErrDown_ratio = TMath::Sqrt(yErr2Down_ratio)*y_ratio;

    graphRatio->SetPoint(iPoint, x_ratio, y_ratio - 1.);
    graphRatio->SetPointError(iPoint, xErrDown_ratio, xErrUp_ratio, yErrDown_ratio, yErrUp_ratio);
  }
  
  graphRatio->SetLineColor(numerator->GetLineColor());
  graphRatio->SetLineWidth(numerator->GetLineWidth());
  graphRatio->SetMarkerColor(numerator->GetMarkerColor());
  graphRatio->SetMarkerStyle(numerator->GetMarkerStyle());
  graphRatio->SetMarkerSize(numerator->GetMarkerSize());

  return graphRatio;
}

void showGraphs(double canvasSizeX, double canvasSizeY,
		TGraph* graph_ref, const std::string& legendEntry_ref,
		TGraph* graph2, const std::string& legendEntry2,
		TGraph* graph3, const std::string& legendEntry3,
		TGraph* graph4, const std::string& legendEntry4,
		TGraph* graph5, const std::string& legendEntry5,
		TGraph* graph6, const std::string& legendEntry6,
		double xMin, double xMax, unsigned numBinsX, const std::string& xAxisTitle, double xAxisOffset,
		bool useLogScale, double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset,
		double legendX0, double legendY0, 
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

  int colors[6] = { kBlack, kGreen - 6, kBlue - 7,  kMagenta - 7, kCyan - 6, kRed - 6 };
  int markerStyles[6] = { 24, 25, 20, 21, 22, 23 };
  int markerSizes[6] = { 1, 1, 1, 1, 1, 1 };

  TLegend* legend = new TLegend(legendX0, legendY0, legendX0 + 0.44, legendY0 + 0.20, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);

  TH1* dummyHistogram_top = new TH1D("dummyHistogram_top", "dummyHistogram_top", numBinsX, xMin, xMax);
  dummyHistogram_top->SetTitle("");
  dummyHistogram_top->SetStats(false);
  dummyHistogram_top->SetMinimum(yMin);
  dummyHistogram_top->SetMaximum(yMax);

  TAxis* xAxis_top = dummyHistogram_top->GetXaxis();
  xAxis_top->SetTitle(xAxisTitle.data());
  xAxis_top->SetTitleSize(0.045);
  xAxis_top->SetTitleOffset(xAxisOffset);

  TAxis* yAxis_top = dummyHistogram_top->GetYaxis();
  yAxis_top->SetTitle(yAxisTitle.data());
  yAxis_top->SetTitleSize(0.045);
  yAxis_top->SetTitleOffset(yAxisOffset);

  dummyHistogram_top->Draw("axis");

  graph_ref->SetLineColor(colors[0]);
  graph_ref->SetLineWidth(2);
  graph_ref->SetMarkerColor(colors[0]);
  graph_ref->SetMarkerStyle(markerStyles[0]);
  graph_ref->SetMarkerSize(markerSizes[0]);
  graph_ref->Draw("p");
  legend->AddEntry(graph_ref, legendEntry_ref.data(), "p");

  if ( graph2 ) {
    graph2->SetLineColor(colors[1]);
    graph2->SetLineWidth(2);
    graph2->SetMarkerColor(colors[1]);
    graph2->SetMarkerStyle(markerStyles[1]);
    graph2->SetMarkerSize(markerSizes[1]);
    graph2->Draw("p");
    legend->AddEntry(graph2, legendEntry2.data(), "p");
  }
  
  if ( graph3 ) {
    graph3->SetLineColor(colors[2]);
    graph3->SetLineWidth(2);
    graph3->SetMarkerColor(colors[2]);
    graph3->SetMarkerStyle(markerStyles[2]);
    graph3->SetMarkerSize(markerSizes[2]);
    graph3->Draw("p");
    legend->AddEntry(graph3, legendEntry3.data(), "p");
  }

  if ( graph4 ) {
    graph4->SetLineColor(colors[3]);
    graph4->SetLineWidth(2);
    graph4->SetMarkerColor(colors[3]);
    graph4->SetMarkerStyle(markerStyles[3]);
    graph4->SetMarkerSize(markerSizes[3]);
    graph4->Draw("p");
    legend->AddEntry(graph4, legendEntry4.data(), "p");
  }

  if ( graph5 ) {
    graph5->SetLineColor(colors[4]);
    graph5->SetLineWidth(2);
    graph5->SetMarkerColor(colors[4]);
    graph5->SetMarkerStyle(markerStyles[4]);
    graph5->SetMarkerSize(markerSizes[4]);
    graph5->Draw("p");
    legend->AddEntry(graph5, legendEntry5.data(), "p");
  }

  if ( graph6 ) {
    graph6->SetLineColor(colors[5]);
    graph6->SetLineWidth(2);
    graph6->SetMarkerColor(colors[5]);
    graph6->SetMarkerStyle(markerStyles[5]);
    graph6->SetMarkerSize(markerSizes[5]);
    graph6->Draw("p");
    legend->AddEntry(graph6, legendEntry6.data(), "p");
  }
  
  legend->Draw();

  canvas->cd();
  bottomPad->Draw();
  bottomPad->cd();

  TH1* dummyHistogram_bottom = new TH1D("dummyHistogram_bottom", "dummyHistogram_bottom", numBinsX, xMin, xMax);
  dummyHistogram_bottom->SetTitle("");
  dummyHistogram_bottom->SetStats(false);
  dummyHistogram_bottom->SetMinimum(-0.25);
  dummyHistogram_bottom->SetMaximum(+0.25);

  TAxis* xAxis_bottom = dummyHistogram_bottom->GetXaxis();
  xAxis_bottom->SetTitle(xAxis_top->GetTitle());
  xAxis_bottom->SetLabelColor(1);
  xAxis_bottom->SetTitleColor(1);
  xAxis_bottom->SetTitleOffset(1.20);
  xAxis_bottom->SetTitleSize(0.08);
  xAxis_bottom->SetLabelOffset(0.02);
  xAxis_bottom->SetLabelSize(0.08);
  xAxis_bottom->SetTickLength(0.055);
  
  TAxis* yAxis_bottom = dummyHistogram_bottom->GetYaxis();
  yAxis_bottom->SetTitle("#frac{Embedding - Z/#gamma^{*} #rightarrow #tau #tau}{Z/#gamma^{*} #rightarrow #tau #tau}");
  yAxis_bottom->SetTitleOffset(0.70);
  yAxis_bottom->SetNdivisions(505);
  yAxis_bottom->CenterTitle();
  yAxis_bottom->SetTitleSize(0.08);
  yAxis_bottom->SetLabelSize(0.08);
  yAxis_bottom->SetTickLength(0.04); 

  dummyHistogram_bottom->Draw("axis");
  
  TGraph* graph2_div_ref = 0;
  if ( graph2 ) {
    std::string graphName2_div_ref = std::string(graph2->GetName()).append("_div_").append(graph_ref->GetName());
    graph2_div_ref = compRatioGraph(graphName2_div_ref, graph2, graph_ref);
    graph2_div_ref->Draw("p");
  }

  TGraph* graph3_div_ref = 0;
  if ( graph3 ) {
    std::string graphName3_div_ref = std::string(graph3->GetName()).append("_div_").append(graph_ref->GetName());
    graph3_div_ref = compRatioGraph(graphName3_div_ref, graph3, graph_ref);
    graph3_div_ref->Draw("p");
  }

  TGraph* graph4_div_ref = 0;
  if ( graph4 ) {
    std::string graphName4_div_ref = std::string(graph4->GetName()).append("_div_").append(graph_ref->GetName());
    graph4_div_ref = compRatioGraph(graphName4_div_ref, graph4, graph_ref);
    graph4_div_ref->Draw("p");
  }

  TGraph* graph5_div_ref = 0;
  if ( graph5 ) {
    std::string graphName5_div_ref = std::string(graph5->GetName()).append("_div_").append(graph_ref->GetName());
    graph5_div_ref = compRatioGraph(graphName5_div_ref, graph5, graph_ref);
    graph5_div_ref->Draw("p");
  }
 
  TGraph* graph6_div_ref = 0;
  if ( graph6 ) {
    std::string graphName6_div_ref = std::string(graph6->GetName()).append("_div_").append(graph_ref->GetName());
    graph6_div_ref = compRatioGraph(graphName6_div_ref, graph6, graph_ref);
    graph6_div_ref->Draw("p");
  }
    
  canvas->Update();
  size_t idx = outputFileName.find_last_of('.');
  std::string outputFileName_plot = std::string(outputFileName, 0, idx);
  if ( useLogScale ) outputFileName_plot.append("_log");
  else outputFileName_plot.append("_linear");
  if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
  canvas->Print(std::string(outputFileName_plot).append(".png").data());
  canvas->Print(std::string(outputFileName_plot).append(".pdf").data());
  canvas->Print(std::string(outputFileName_plot).append(".root").data());
  
  delete legend;
  delete graph2_div_ref;
  delete graph3_div_ref;
  delete graph4_div_ref;
  delete graph5_div_ref;
  delete graph6_div_ref;
  delete dummyHistogram_top;
  delete topPad;
  delete dummyHistogram_bottom;
  delete bottomPad;
  delete canvas;  
}
//-------------------------------------------------------------------------------

void makeEmbeddingValidationPlots()
{
//--- stop ROOT from keeping references to all histograms
  TH1::AddDirectory(false);

//--- suppress the output canvas 
  gROOT->SetBatch(true);

  std::string inputFilePath = "/data1/veelken/tmp/EmbeddingValidation/";
  //std::string inputFilePath = "/tmp/veelken/";

  //std::string channel = "etau";
  std::string channel = "mutau";
  //std::string channel = "emu";
  
  std::map<std::string, std::string> inputFileNames;
  std::vector<std::string> processes;
  std::string dqmDirectory_afterGenCuts;
  std::string dqmDirectory_afterGenAndRecCuts;
  if ( channel == "etau" ) {
    inputFileNames["simDYtoTauTau"]                         = "validateMCEmbedding_simDYtoTauTau_etau_all_v2_5_2.root";
    inputFileNames["simDYtoMuMu_recEmbedding_HCP"]          = "validateMCEmbedding_simDYtoMuMu_embedEqPF_cleanEqPF_replaceRecMuons_by_etau_HCP_all_v2_5_2_kineReweighted.root";
    inputFileNames["simDYtoMuMu_genEmbedding_wTauSpinner"]  = "validateMCEmbedding_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceGenMuons_by_ePtGt20tauPtGt18_embedAngleEq90_noPolarization_wTauSpinner_all_v2_5_2.root";
    inputFileNames["simDYtoMuMu_recEmbedding_wTauSpinner"]  = "validateMCEmbedding_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceRecMuons_by_ePtGt20tauPtGt18_embedAngleEq90_noPolarization_wTauSpinner_all_v2_5_2.root";
    //inputFileNames["data2012runA_recEmbedding_wTauSpinner"] = "validateMCEmbedding_data2012runA_embedEqRH_cleanEqDEDX_replaceRecMuons_by_etau_embedAngleEq90_noPolarization_wTauSpinner_all_v2_1_0_kineReweighted.root";
    processes.push_back("simDYtoTauTau");
    processes.push_back("simDYtoMuMu_recEmbedding_HCP");
    //processes.push_back("simDYtoMuMu_recEmbedding_HCP_BLAH"); // CV: place-holder to avoid that colors change
    processes.push_back("simDYtoMuMu_genEmbedding_wTauSpinner");
    processes.push_back("simDYtoMuMu_recEmbedding_wTauSpinner");
    //processes.push_back("data2012runA_recEmbedding_wTauSpinner");
    dqmDirectory_afterGenCuts = "validationAnalyzerAfterGenCutsDR00_etau";
    dqmDirectory_afterGenAndRecCuts = "validationAnalyzerAfterGenAndRecCutsDR00_etau";
  } else if ( channel == "mutau" ) {
    inputFileNames["simDYtoTauTau"]                         = "validateMCEmbedding_simDYtoTauTau_mutau_all_v2_5_2.root";
    inputFileNames["simDYtoMuMu_recEmbedding_HCP"]          = "validateMCEmbedding_simDYtoMuMu_embedEqPF_cleanEqPF_replaceRecMuons_by_mutau_HCP_all_v2_5_2_kineReweighted.root";
    inputFileNames["simDYtoMuMu_genEmbedding_wTauSpinner"]  = "validateMCEmbedding_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceGenMuons_by_muPtGt16tauPtGt18_embedAngleEq90_noPolarization_wTauSpinner_all_v2_5_2.root";
    //inputFileNames["simDYtoMuMu_recEmbedding_wTauSpinner"]  = "validateMCEmbedding_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceRecMuons_by_muPtGt16tauPtGt18_embedAngleEq90_noPolarization_wTauSpinner_all_v2_5_2.root";
    inputFileNames["simDYtoMuMu_recEmbedding_wTauSpinner"]  = "validateMCEmbedding_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceRecMuons_by_muPtGt16tauPtGt18_embedAngleEqMinus90_noPolarization_wTauSpinner_all_v2_6_5.root";
    //inputFileNames["data2012runA_recEmbedding_wTauSpinner"] = "validateMCEmbedding_data2012runA_embedEqRH_cleanEqDEDX_replaceRecMuons_by_mutau_embedAngleEq90_noPolarization_wTauSpinner_all_v2_1_0_kineReweighted.root";
    processes.push_back("simDYtoTauTau");
    processes.push_back("simDYtoMuMu_recEmbedding_HCP");
    //processes.push_back("simDYtoMuMu_recEmbedding_HCP_BLAH"); // CV: place-holder to avoid that colors change
    processes.push_back("simDYtoMuMu_genEmbedding_wTauSpinner");
    processes.push_back("simDYtoMuMu_recEmbedding_wTauSpinner");
    //processes.push_back("data2012runA_recEmbedding_wTauSpinner");
    dqmDirectory_afterGenCuts = "validationAnalyzerAfterGenCutsDR00_mutau";
    dqmDirectory_afterGenAndRecCuts = "validationAnalyzerAfterGenAndRecCutsDR00_mutau";
  } else if ( channel == "emu" ) {
    inputFileNames["simDYtoTauTau"]                         = "validateMCEmbedding_simDYtoTauTau_emu_all_v2_1_5_kineReweighted.root";
    inputFileNames["simDYtoMuMu_genEmbedding_wTauSpinner"]  = "validateMCEmbedding_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceGenMuons_by_emu_embedAngleEq90_noPolarization_wTauSpinner_all_v2_1_6_kineReweighted.root";
    inputFileNames["simDYtoMuMu_recEmbedding_wTauSpinner"]  = "validateMCEmbedding_simDYtoMuMu_embedEqRH_cleanEqDEDX_replaceRecMuons_by_emu_embedAngleEq90_noPolarization_wTauSpinner_all_v2_1_6_kineReweighted.root";
    processes.push_back("simDYtoTauTau");
    processes.push_back("simDYtoMuMu_genEmbedding_wTauSpinner");
    processes.push_back("simDYtoMuMu_recEmbedding_wTauSpinner");
    dqmDirectory_afterGenCuts = "validationAnalyzerAfterGenCutsDR00_emu";
    dqmDirectory_afterGenAndRecCuts = "validationAnalyzerAfterGenAndRecCutsDR00_emu";
  } else {
    std::cout << "Invalid channel = " << channel << " !!" << std::endl;
    assert(0);
  }
  
  std::map<std::string, std::string> legendEntries;
  legendEntries["simDYtoTauTau"]                         = "gen. Z/#gamma^{*} #rightarrow #tau #tau";
  legendEntries["simDYtoMuMu_recEmbedding_HCP"]          = "PF-Embedding, rec.#mu";
  legendEntries["simDYtoMuMu_genEmbedding_wTauSpinner"]  = "recHit-Embedding, gen.#mu"; 
  legendEntries["simDYtoMuMu_recEmbedding_wTauSpinner"]  = "recHit-Embedding, rec.#mu"; 
  legendEntries["data2012runA_recEmbedding_wTauSpinner"] = "Data 2012 run A"; 

  std::vector<plotEntryType_distribution> distributionsToPlot;
  distributionsToPlot.push_back(plotEntryType_distribution(
    "vertexX", "vertexX", 
    dqmDirectory_afterGenAndRecCuts, "theRecVertexX", -1., +1., 200, "x_{Vtx}", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "vertexY", "vertexY", 
    dqmDirectory_afterGenAndRecCuts, "theRecVertexY", -1., +1., 200, "y_{Vtx}", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "vertexZ", "vertexZ", 
    dqmDirectory_afterGenAndRecCuts, "theRecVertexZ", -25., +25., 500, "z_{Vtx}", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "numVertices", "numVertices", 
    dqmDirectory_afterGenAndRecCuts, "numRecVertices", -0.5, +34.5, 35, "N_{Vtx}", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "beamSpotX", "beamSpotX", 
    dqmDirectory_afterGenAndRecCuts, "beamSpotX", -1., +1., 200, "x_{BS}", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "beamSpotY", "beamSpotY", 
    dqmDirectory_afterGenAndRecCuts, "beamSpotY", -1., +1., 200, "y_{BS}", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));

  distributionsToPlot.push_back(plotEntryType_distribution(
    "numGlobalMuons", "numGlobalMuons", 
    dqmDirectory_afterGenAndRecCuts, "numGlobalMuons", -0.5, +9.5, 10, "Num. global Muons", 1.2, 0., 1., "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "numStandAloneMuons", "numStandAloneMuons", 
    dqmDirectory_afterGenAndRecCuts, "numStandAloneMuons", -0.5, +9.5, 10, "Num. stand-alone Muons", 1.2, 0., 1., "a.u.", 1.2, false));

  distributionsToPlot.push_back(plotEntryType_distribution(
    "numTracksPtGt5", "numTracksPtGt5", 
    dqmDirectory_afterGenAndRecCuts, "numTracksPtGt5", -0.5, +49.5, 50, "Num. Tracks of P_{T} > 5 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "numTracksPtGt10", "numTracksPtGt10", 
    dqmDirectory_afterGenAndRecCuts, "numTracksPtGt10", -0.5, +34.5, 50, "Num. Tracks of P_{T} > 10 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "numTracksPtGt20", "numTracksPtGt20", 
    dqmDirectory_afterGenAndRecCuts, "numTracksPtGt20", -0.5, +19.5, 50, "Num. Tracks of P_{T} > 20 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "numTracksPtGt30", "numTracksPtGt30", 
    dqmDirectory_afterGenAndRecCuts, "numTracksPtGt30", -0.5, +19.5, 50, "Num. Tracks of P_{T} > 30 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "numTracksPtGt40", "numTracksPtGt40", 
    dqmDirectory_afterGenAndRecCuts, "numTracksPtGt40", -0.5, +19.5, 50, "Num. Tracks of P_{T} > 40 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  
  distributionsToPlot.push_back(plotEntryType_distribution(
    "numJetsRawPtGt20", "numJetsRawPtGt20", 
    dqmDirectory_afterGenAndRecCuts, "numJetsRawPtGt20", -0.5, +19.5, 50, "Num. Jets of P_{T}^{raw} > 20 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "numJetsCorrPtGt20", "numJetsCorrPtGt20", 
    dqmDirectory_afterGenAndRecCuts, "numJetsCorrPtGt20", -0.5, +19.5, 50, "Num. Jets of P_{T}^{corr} > 20 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "numJetsRawPtGt30", "numJetsRawPtGt30", 
    dqmDirectory_afterGenAndRecCuts, "numJetsRawPtGt30", -0.5, +19.5, 50, "Num. Jets of P_{T}^{raw} > 30 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "numJetsCorrPtGt30", "numJetsCorrPtGt30", 
    dqmDirectory_afterGenAndRecCuts, "numJetsCorrPtGt30", -0.5, +19.5, 50, "Num. Jets of P_{T}^{corr} > 30 GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));

  distributionsToPlot.push_back(plotEntryType_distribution(
    "genTau1Pt", "genTau1Pt", 
    dqmDirectory_afterGenAndRecCuts, "genTau1Pt", 0., 250., 50, "P_{T}^{1}", 1.3, 0., 0.32,  "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genTau1Pt", "genTau1Pt", 
    dqmDirectory_afterGenAndRecCuts, "genTau1Pt", 0., 250., 50, "P_{T}^{1}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genTau1Eta", "genTau1Eta", 
    dqmDirectory_afterGenAndRecCuts, "genTau1Eta", -2.5, +2.5, 50, "#eta_{1}", 1.3, 0., 0.32,  "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg1Pt", "genLeg1Pt", 
    dqmDirectory_afterGenAndRecCuts, "genLeg1Pt", 0., 250., 50, "P_{T}^{1}", 1.3, 0., 0.32,  "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg1Pt", "genLeg1Pt", 
    dqmDirectory_afterGenAndRecCuts, "genLeg1Pt", 0., 250., 50, "P_{T}^{1}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg1X", "genLeg1X", 
    dqmDirectory_afterGenAndRecCuts, "genLeg1X", 0., 1., 100, "X_{1}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg1XforGenLeg2X0_00to0_25", "genLeg1XforGenLeg2X0_00to0_25", 
    dqmDirectory_afterGenAndRecCuts, "genLeg1XforGenLeg2X0_00to0_25", 0., 1., 20, "X_{1} (X_{2} < 0.25)", 1.2, 0., 0.12, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg1XforGenLeg2X0_25to0_50", "genLeg1XforGenLeg2X0_25to0_50", 
    dqmDirectory_afterGenAndRecCuts, "genLeg1XforGenLeg2X0_25to0_50", 0., 1., 20, "X_{1} (0.25 < X_{2} < 0.50)", 1.2, 0., 0.12, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg1XforGenLeg2X0_50to0_75", "genLeg1XforGenLeg2X0_50to0_75", 
    dqmDirectory_afterGenAndRecCuts, "genLeg1XforGenLeg2X0_50to0_75", 0., 1., 20, "X_{1} (0.50 < X_{2} < 0.75)", 1.2, 0., 0.12, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg1XforGenLeg2X0_75to1_00", "genLeg1XforGenLeg2X0_75to1_00", 
    dqmDirectory_afterGenAndRecCuts, "genLeg1XforGenLeg2X0_75to1_00", 0., 1., 20, "X_{1} (X_{2} > 0.75)", 1.2, 0., 0.12, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(							   
    "genTau2Pt", "genTau2Pt", 
    dqmDirectory_afterGenAndRecCuts, "genTau2Pt", 0., 250., 50, "P_{T}^{2}", 1.3, 0., 0.32,  "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genTau2Pt", "genTau2Pt", 
    dqmDirectory_afterGenAndRecCuts, "genTau2Pt", 0., 250., 50, "P_{T}^{2}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genTau2Eta", "genTau2Eta", 
    dqmDirectory_afterGenAndRecCuts, "genTau2Eta", -2.5, +2.5, 50, "#eta_{2}", 1.3, 0., 0.32,  "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(							   
    "genLeg2Pt", "genLeg2Pt", 
    dqmDirectory_afterGenAndRecCuts, "genLeg2Pt", 0., 250., 50, "P_{T}^{2}", 1.3, 0., 0.32,  "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg2Pt", "genLeg2Pt", 
    dqmDirectory_afterGenAndRecCuts, "genLeg2Pt", 0., 250., 50, "P_{T}^{2}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg2X", "genLeg2X", 
    dqmDirectory_afterGenAndRecCuts, "genLeg2X", 0., 1., 100, "X_{2}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg2XforGenLeg1X0_00to0_25", "genLeg2XforGenLeg1X0_00to0_25", 
    dqmDirectory_afterGenAndRecCuts, "genLeg2XforGenLeg1X0_00to0_25", 0., 1., 20, "X_{2} (X_{1} < 0.25)", 1.2, 0., 0.12, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg2XforGenLeg1X0_25to0_50", "genLeg2XforGenLeg1X0_25to0_50", 
    dqmDirectory_afterGenAndRecCuts, "genLeg2XforGenLeg1X0_25to0_50", 0., 1., 20, "X_{2} (0.25 < X_{1} < 0.50)", 1.2, 0., 0.12, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg2XforGenLeg1X0_50to0_75", "genLeg2XforGenLeg1X0_50to0_75", 
    dqmDirectory_afterGenAndRecCuts, "genLeg2XforGenLeg1X0_50to0_75", 0., 1., 20, "X_{2} (0.50 < X_{1} < 0.75)", 1.2, 0., 0.12, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genLeg2XforGenLeg1X0_75to1_00", "genLeg2XforGenLeg1X0_75to1_00", 
    dqmDirectory_afterGenAndRecCuts, "genLeg2XforGenLeg1X0_75to1_00", 0., 1., 20, "X_{2} (X_{1} > 0.75)", 1.2, 0., 0.12, "a.u.", 1.2, false));

  distributionsToPlot.push_back(plotEntryType_distribution(
    "genElectronEta", "genElectronEta", 
    dqmDirectory_afterGenAndRecCuts, "goodElectronDistributions/genLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.07, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genElectronPt", "genElectronPt",  
    dqmDirectory_afterGenAndRecCuts, "goodElectronDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genElectronPt", "genElectronPt",  
    dqmDirectory_afterGenAndRecCuts, "goodElectronDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recElectronEta", "recElectronEta", 
    dqmDirectory_afterGenAndRecCuts, "goodElectronDistributions/recLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.07, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recElectronPt", "recElectronPt",  
    dqmDirectory_afterGenAndRecCuts, "goodElectronDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recElectronPt", "recElectronPt",  
    dqmDirectory_afterGenAndRecCuts, "goodElectronDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recMinusGenElectronPt", "recMinusGenElectronPt",  
    dqmDirectory_afterGenAndRecCuts, "goodElectronDistributions/recMinusGenLeptonPt", -50., +25., 75, "P_{T}^{#mu,rec} - P_{T}^{#mu,gen}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
/* 
  std::vector<std::string> electronSelections;
  electronSelections.push_back("gsfElectron");
  electronSelections.push_back("gsfPFElectronBarrelTrkSeed");
  electronSelections.push_back("gsfNonPFElectronBarrelTrkSeed");
  electronSelections.push_back("gsfPFElectronBarrelECALseed");
  electronSelections.push_back("gsfNonPFElectronBarrelECALseed");
  electronSelections.push_back("gsfPFElectronEndcapTrkSeed");
  electronSelections.push_back("gsfNonPFElectronEndcapTrkSeed");
  electronSelections.push_back("gsfPFElectronEndcapECALseed");
  electronSelections.push_back("gsfNonPFElectronEndcapECALseed");
  electronSelections.push_back("selElectronMVA");
  electronSelections.push_back("selElectronAntiConv");
  std::map<std::string, std::string> dqmSubDirectories_distribution;
  dqmSubDirectories_distribution["gsfElectron"]                    = "genMatchedPatElectronDistributions";
  dqmSubDirectories_distribution["gsfPFElectronBarrelTrkSeed"]     = "genMatchedPatPFElectronBarrelTrkSeedDistributions";
  dqmSubDirectories_distribution["gsfNonPFElectronBarrelTrkSeed"]  = "genMatchedPatNonPFElectronBarrelTrkSeedDistributions";
  dqmSubDirectories_distribution["gsfPFElectronBarrelECALseed"]    = "genMatchedPatPFElectronBarrelECALseedDistributions";
  dqmSubDirectories_distribution["gsfNonPFElectronBarrelECALseed"] = "genMatchedPatNonPFElectronBarrelECALseedDistributions";
  dqmSubDirectories_distribution["gsfPFElectronEndcapTrkSeed"]     = "genMatchedPatPFElectronEndcapTrkSeedDistributions";
  dqmSubDirectories_distribution["gsfNonPFElectronEndcapTrkSeed"]  = "genMatchedPatNonPFElectronEndcapTrkSeedDistributions";
  dqmSubDirectories_distribution["gsfPFElectronEndcapECALseed"]    = "genMatchedPatPFElectronEndcapECALseedDistributions";
  dqmSubDirectories_distribution["gsfNonPFElectronEndcapECALseed"] = "genMatchedPatNonPFElectronEndcapECALseedDistributions";
  dqmSubDirectories_distribution["selElectronMVA"]                 = "selectedElectronIdMVAdistributions";
  dqmSubDirectories_distribution["selElectronAntiConv"]            = "selectedElectronConversionVetoDistributions";
  for ( std::vector<std::string>::const_iterator electronSelection = electronSelections.begin();
	electronSelection != electronSelections.end(); ++electronSelection ) {
    const std::string& dqmSubDirectory = dqmSubDirectories_distribution[*electronSelection];
    if ( dqmSubDirectory == "" ) continue;
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sMVAptLt20AbsEtaLt0_8", electronSelection->data()), Form("%sMVAptLt20AbsEtaLt0_8", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/MVAptLt20AbsEtaLt0_8", dqmSubDirectory.data()), -1.005, +1.005, 201, "MVA", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sMVAptLt20AbsEta0_8to1_479", electronSelection->data()), Form("%sMVAptLt20AbsEta0_8to1_479", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/MVAptLt20AbsEta0_8to1_479", dqmSubDirectory.data()), -1.005, +1.005, 201, "MVA", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sMVAptLt20AbsEtaGt1_479", electronSelection->data()), Form("%sMVAptLt20AbsEtaGt1_479", electronSelection->data()),
      dqmDirectory_afterGenAndRecCuts, Form("%s/MVAptLt20AbsEtaGt1_479", dqmSubDirectory.data()), -1.005, +1.005, 201, "MVA", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sMVAptGt20AbsEtaLt0_8", electronSelection->data()), Form("%sMVAptGt20AbsEtaLt0_8", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/MVAptGt20AbsEtaLt0_8", dqmSubDirectory.data()), -1.005, +1.005, 201, "MVA", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sMVAptGt20AbsEta0_8to1_479", electronSelection->data()), Form("%sMVAptGt20AbsEta0_8to1_479", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/MVAptGt20AbsEta0_8to1_479", dqmSubDirectory.data()), -1.005, +1.005, 201, "MVA", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sMVAptGt20AbsEtaGt1_479", electronSelection->data()), Form("%sMVAptGt20AbsEtaGt1_479", electronSelection->data()),
      dqmDirectory_afterGenAndRecCuts, Form("%s/MVAptGt20AbsEtaGt1_479", dqmSubDirectory.data()), -1.005, +1.005, 201, "MVA", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sFBrem", electronSelection->data()), Form("%sFBrem", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/fBrem", dqmSubDirectory.data()), -1., +1., 200, "f_{Brem}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sKFchi2", electronSelection->data()), Form("%sKFchi2", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/kfChi2", dqmSubDirectory.data()), 0., 10., 100, "#Chi^{2}_{KF}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sNumKFhits", electronSelection->data()), Form("%sNumKFhits", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/numKFhits", dqmSubDirectory.data()), -0.5, +19.5, 20, "N_{hit}^{KF}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sGSFchi2", electronSelection->data()), Form("%sGSFchi2", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/gsfChi2", dqmSubDirectory.data()), 0., 25., 250, "#Chi^{2}_{GSF}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sDEta", electronSelection->data()), Form("%sDEta", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/dEta", dqmSubDirectory.data()), 0., 0.07, 70, "#Delta#eta", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
     distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sDPhi", electronSelection->data()), Form("%sDPhi", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/dPhi", dqmSubDirectory.data()), 0., 0.50, 200, "#Delta#phi", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sDEtaCalo", electronSelection->data()), Form("%sDEtaCalo", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/dEtaCalo", dqmSubDirectory.data()), 0., 0.21, 210, "#Delta#eta(calo)", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sClusterSee", electronSelection->data()), Form("%sClusterSee", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/See", dqmSubDirectory.data()), 0., 0.10, 100, "S_{#eta#eta}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sClusterSpp", electronSelection->data()), Form("%sClusterSpp", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/Spp", dqmSubDirectory.data()), 0., 0.10, 100, "S_{#phi#phi}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sClusterEtaWidth", electronSelection->data()), Form("%sClusterEtaWidth", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/etaWidth", dqmSubDirectory.data()), 0., 0.10, 100, "width_{#eta}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sClusterPhiWidth", electronSelection->data()), Form("%sClusterPhiWidth", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/phiWidth", dqmSubDirectory.data()), 0., 0.25, 100, "width_{#phi}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sClusterOneMinusE1x5E5x5", electronSelection->data()), Form("%sClusterOneMinusE1x5E5x5", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/oneMinusE1x5E5x5", dqmSubDirectory.data()), 0., 1., 100, "1 - E1x5/E5x5", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sClusterR9", electronSelection->data()), Form("%sClusterR9", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/R9", dqmSubDirectory.data()), 0., 1.1, 110, "R9", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sClusterHoE", electronSelection->data()), Form("%sClusterHoE", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/HoE", dqmSubDirectory.data()), 0., 1., 100, "H/E", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sEoP", electronSelection->data()), Form("%sEoP", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/EoP", dqmSubDirectory.data()), 0., 5., 100, "E/P", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sIoEmIoP", electronSelection->data()), Form("%sIoEmIoP", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/IoEmIoP", dqmSubDirectory.data()), 0., 0.5, 50, "IoEmIoP", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sEleEoPout", electronSelection->data()), Form("%sEleEoPout", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/EleEoPout", dqmSubDirectory.data()), 0., 5., 100, "E/P_{out}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sPreShowerOverRaw", electronSelection->data()), Form("%sPreShowerOverRaw", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/PreShowerOverRaw", dqmSubDirectory.data()), 0., 0.5, 50, "E_{PS}/E", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sTrackD0", electronSelection->data()), Form("%sTrackD0", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/D0", dqmSubDirectory.data()), 0., 0.1, 100, "dIP_{xy}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sTrackIP3d", electronSelection->data()), Form("%sTrackIP3d", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/IP3d", dqmSubDirectory.data()), 0., 0.1, 100, "dIP_{3d}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sEta", electronSelection->data()), Form("%sEta", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/eta", dqmSubDirectory.data()), -2.5, +2.5, 50, "#eta_{e}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sPt", electronSelection->data()), Form("%sPt", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/pt", dqmSubDirectory.data()), 0., 250., 50, "P_{T}^{e}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sCharge", electronSelection->data()), Form("%sCharge", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/charge", dqmSubDirectory.data()), -1.5, +1.5, 3, "Charge", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
    distributionsToPlot.push_back(plotEntryType_distribution(
      Form("%sFlags", electronSelection->data()), Form("%sFlags", electronSelection->data()), 
      dqmDirectory_afterGenAndRecCuts, Form("%s/flags", dqmSubDirectory.data()), -0.5, +11.5, 12, "Flags", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  }
 */
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genIsoElectronEta", "genIsoElectronEta", 
    dqmDirectory_afterGenAndRecCuts, "goodIsoElectronDistributions/genLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.07, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genIsoElectronPt", "genIsoElectronPt",  
    dqmDirectory_afterGenAndRecCuts, "goodIsoElectronDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genIsoElectronPt", "genIsoElectronPt",  
    dqmDirectory_afterGenAndRecCuts, "goodIsoElectronDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recIsoElectronEta", "recIsoElectronEta", 
    dqmDirectory_afterGenAndRecCuts, "goodIsoElectronDistributions/recLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.07, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recIsoElectronPt", "recIsoElectronPt",  
    dqmDirectory_afterGenAndRecCuts, "goodIsoElectronDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recIsoElectronPt", "recIsoElectronPt",  
    dqmDirectory_afterGenAndRecCuts, "goodIsoElectronDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recMinusGenIsoElectronPt", "recMinusGenIsoElectronPt",  
    dqmDirectory_afterGenAndRecCuts, "goodIsoElectronDistributions/recMinusGenLeptonPt", -50., +25., 75, "P_{T}^{#mu,rec} - P_{T}^{#mu,gen}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));

  distributionsToPlot.push_back(plotEntryType_distribution(
    "genMuonEta", "genMuonEta", 
    dqmDirectory_afterGenAndRecCuts, "goodMuonDistributions/genLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genMuonPt", "genMuonPt",  
    dqmDirectory_afterGenAndRecCuts, "goodMuonDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genMuonPt", "genMuonPt",  
    dqmDirectory_afterGenAndRecCuts, "goodMuonDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recMuonEta", "recMuonEta", 
    dqmDirectory_afterGenAndRecCuts, "goodMuonDistributions/recLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recMuonPt", "recMuonPt",  
    dqmDirectory_afterGenAndRecCuts, "goodMuonDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recMuonPt", "recMuonPt",  
    dqmDirectory_afterGenAndRecCuts, "goodMuonDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recMinusGenMuonPt", "recMinusGenMuonPt",  
    dqmDirectory_afterGenAndRecCuts, "goodMuonDistributions/recMinusGenLeptonPt", -50., +25., 75, "P_{T}^{#mu,rec} - P_{T}^{#mu,gen}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));

  distributionsToPlot.push_back(plotEntryType_distribution(
    "genIsoMuonEta", "genIsoMuonEta", 
    dqmDirectory_afterGenAndRecCuts, "goodIsoMuonDistributions/genLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genIsoMuonPt", "genIsoMuonPt",  
    dqmDirectory_afterGenAndRecCuts, "goodIsoMuonDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genIsoMuonPt", "genIsoMuonPt",  
    dqmDirectory_afterGenAndRecCuts, "goodIsoMuonDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recIsoMuonEta", "recIsoMuonEta", 
    dqmDirectory_afterGenAndRecCuts, "goodIsoMuonDistributions/recLeptonEta", -2.5, +2.5, 50, "#eta_{#mu}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recIsoMuonPt", "recIsoMuonPt",  
    dqmDirectory_afterGenAndRecCuts, "goodIsoMuonDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recIsoMuonPt", "recIsoMuonPt",  
    dqmDirectory_afterGenAndRecCuts, "goodIsoMuonDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#mu}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recMinusGenIsoMuonPt", "recMinusGenIsoMuonPt",  
    dqmDirectory_afterGenAndRecCuts, "goodIsoMuonDistributions/recMinusGenLeptonPt", -50., +25., 75, "P_{T}^{#mu,rec} - P_{T}^{#mu,gen}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));

  distributionsToPlot.push_back(plotEntryType_distribution(
    "genTauEta", "genTauEta", 
    dqmDirectory_afterGenAndRecCuts, "selectedTauDistributions/genLeptonEta", -2.5, +2.5, 50, "#eta_{#tau}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genTauPt", "genTauPt",  
    dqmDirectory_afterGenAndRecCuts, "selectedTauDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#tau}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genTauPt", "genTauPt",  
    dqmDirectory_afterGenAndRecCuts, "selectedTauDistributions/genLeptonPt", 0., 250., 50, "P_{T}^{#tau}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genTauDecayMode", "genTauDecayMode", 
    dqmDirectory_afterGenAndRecCuts, "selectedTauDistributions/genTauDecayMode", -1.5, +19.5, 21, "gen. Tau Decay Mode", 1.2, 0., 0.60, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recTauEta", "recTauEta", 
    dqmDirectory_afterGenAndRecCuts, "selectedTauDistributions/recLeptonEta", -2.5, +2.5, 50, "#eta_{#tau}", 1.2, 0., 0.05, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recTauPt", "recTauPt",  
    dqmDirectory_afterGenAndRecCuts, "selectedTauDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#tau}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recTauPt",  "recTauPt",  
    dqmDirectory_afterGenAndRecCuts, "selectedTauDistributions/recLeptonPt", 0., 250., 50, "P_{T}^{#tau}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recMinusGenTauPt", "recMinusGenTauPt",  
    dqmDirectory_afterGenAndRecCuts, "selectedTauDistributions/recMinusGenLeptonPt", -50., +25., 75 , "P_{T}^{#tau,rec} - P_{T}^{#tau,gen}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recTauDecayMode", "recTauDecayMode", 
    dqmDirectory_afterGenAndRecCuts, "selectedTauDistributions/recTauDecayMode", -1.5, +19.5, 21, "rec. Tau Decay Mode", 1.2, 0., 0.60, "a.u.", 1.2, false));

  distributionsToPlot.push_back(plotEntryType_distribution(
    "genDiTauPtS", "genDiTauPt",  
    dqmDirectory_afterGenAndRecCuts, "genDiTauPt", 0., 50., 50, "P_{T}^{#tau#tau}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genDiTauPtL", "genDiTauPt",  
    dqmDirectory_afterGenAndRecCuts, "genDiTauPt", 0., 250., 50, "P_{T}^{#tau#tau}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genDiTauPtS", "genDiTauPt",  
    dqmDirectory_afterGenAndRecCuts, "genDiTauPt", 0., 50., 50, "P_{T}^{#tau#tau}", 1.3, 1.e-5, 1.e+0, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genDiTauPtL", "genDiTauPt",  
    dqmDirectory_afterGenAndRecCuts, "genDiTauPt", 0., 250., 50, "P_{T}^{#tau#tau}", 1.3, 1.e-5, 1.e+0, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genDiTauMass", "genDiTauMass",  
    dqmDirectory_afterGenAndRecCuts, "genDiTauMass", 20., 250., 46, "M_{#tau#tau}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genDiTauMass", "genDiTauMass",  
    dqmDirectory_afterGenAndRecCuts, "genDiTauMass", 20., 250., 46, "M_{#tau#tau}", 1.3, 1.e-5, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genVisDiTauPt", "genVisDiTauPt",  
    dqmDirectory_afterGenAndRecCuts, "genVisDiTauPt", 0., 250., 50, "P_{T}^{vis}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genVisDiTauPt", "genVisDiTauPt",  
    dqmDirectory_afterGenAndRecCuts, "genVisDiTauPt", 0., 250., 50, "P_{T}^{vis}", 1.3, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genVisDiTauMass", "genVisDiTauMass",  
    dqmDirectory_afterGenAndRecCuts, "genVisDiTauMass", 20., 250., 46, "M_{vis}", 1.3, 0., 0.32, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genVisDiTauMass", "genVisDiTauMass",  
    dqmDirectory_afterGenAndRecCuts, "genVisDiTauMass", 20., 250., 46, "M_{vis}", 1.3, 1.e-6, 1.e+0, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genDeltaPhiLeg1Leg2", "genDeltaPhiLeg1Leg2",  
    dqmDirectory_afterGenAndRecCuts, "genDeltaPhiLeg1Leg2", 0., TMath::Pi(), 36, "#Delta#phi_{12}", 1.3, 1.e-4, 1.e+0, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genDiTauDecayAngle", "genDiTauDecayAngle",  
    dqmDirectory_afterGenAndRecCuts, "genDiTauDecayAngle", 0., TMath::Pi(), 36, "#theta_{GJ}", 1.3, 1.e-4, 1.e+0, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recVisDiTauMass", "recVisDiTauMass",  
    dqmDirectory_afterGenAndRecCuts, "recVisDiTauMass", 20., 250., 46, "M_{#tau#tau}", 1.3, 0., 0.32, "a.u.", 1.2, false));

  distributionsToPlot.push_back(plotEntryType_distribution(
    "genMEt", "genMEt", 
    dqmDirectory_afterGenAndRecCuts, "type1CorrPFMEtDistributions/genMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "genMEt", "genMEt", 
    dqmDirectory_afterGenAndRecCuts, "type1CorrPFMEtDistributions/genMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recCaloMEtNoHF", "recCaloMEtNoHF", 
    dqmDirectory_afterGenAndRecCuts, "rawCaloMEtNoHFdistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recCaloMEtNoHF", "recCaloMEtNoHF", 
    dqmDirectory_afterGenAndRecCuts, "rawCaloMEtNoHFdistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recPFMEt", "recPFMEt", 
    dqmDirectory_afterGenAndRecCuts, "rawPFMEtDistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recPFMEt", "recPFMEt", 
    dqmDirectory_afterGenAndRecCuts, "rawPFMEtDistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recPFMEtTypeIcorrected", "recPFMEtTypeIcorrected", 
    dqmDirectory_afterGenAndRecCuts, "type1CorrPFMEtDistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recPFMEtTypeIcorrected", "recPFMEtTypeIcorrected", 
    dqmDirectory_afterGenAndRecCuts, "type1CorrPFMEtDistributions/recMEtPt", 0., 250., 50, "E_{T}^{miss} / GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recPFMEtTypeIcorrectedMinusGenMEtParlZ", "recPFMEtTypeIcorrectedMinusGenMEtParlZ", 
    dqmDirectory_afterGenAndRecCuts, "type1CorrPFMEtDistributions/recMinusGenMEtParlZ", -100., +100., 50, "#DeltaE_{#parallel}^{miss} / GeV", 1.2, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recPFMEtTypeIcorrectedMinusGenMEtParlZ", "recPFMEtTypeIcorrectedMinusGenMEtParlZ", 
    dqmDirectory_afterGenAndRecCuts, "type1CorrPFMEtDistributions/recMinusGenMEtParlZ", -100., +100., 50, "#DeltaE_{#parallel}^{miss} / GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recPFMEtTypeIcorrectedMinusGenMEtPerpZ", "recPFMEtTypeIcorrectedMinusGenMEtPerpZ", 
    dqmDirectory_afterGenAndRecCuts, "type1CorrPFMEtDistributions/recMinusGenMEtPerpZ", 0., 100., 50, "#DeltaE_{#perp}^{miss} / GeV", 1.2, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "recPFMEtTypeIcorrectedMinusGenMEtPerpZ", "recPFMEtTypeIcorrectedMinusGenMEtPerpZ", 
    dqmDirectory_afterGenAndRecCuts, "type1CorrPFMEtDistributions/recMinusGenMEtPerpZ", 0., 100., 50, "#DeltaE_{#perp}^{miss} / GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "L1ETM", "L1ETM", 
    dqmDirectory_afterGenAndRecCuts, "L1ETM", 0., 250., 50, "L1-E_{T}^{miss} / GeV", 1.2, 0., 0.25, "a.u.", 1.2, false));
  distributionsToPlot.push_back(plotEntryType_distribution(
    "L1ETM", "L1ETM", 
    dqmDirectory_afterGenAndRecCuts, "L1ETM", 0., 250., 50, "L1-E_{T}^{miss} / GeV", 1.2, 1.e-6, 1.e+1, "a.u.", 1.2, true));

  std::map<std::string, TFile*> inputFiles;
  for ( std::map<std::string, std::string>::const_iterator inputFileName = inputFileNames.begin();
	inputFileName != inputFileNames.end(); ++inputFileName ) {
    inputFiles[inputFileName->first] = new TFile(std::string(inputFilePath).append(inputFileName->second).data());
    if ( !inputFiles[inputFileName->first] ) {
      std::cerr << "Failed to open inputFile = " << inputFileName->second << " !!" << std::endl;
      assert(0);
    }
  }

  for ( std::vector<plotEntryType_distribution>::const_iterator plot = distributionsToPlot.begin();
	plot != distributionsToPlot.end(); ++plot ) {
    std::vector<TH1*> histograms_plot(6);
    std::vector<std::string> legendEntries_plot(6);
    unsigned numProcesses = processes.size();
    for ( unsigned iProcess = 0; iProcess < numProcesses; ++iProcess ) {
      const std::string& process = processes[iProcess];
      histograms_plot[iProcess] = getHistogram(inputFiles[process], "", plot->meName_);
      TH1* histogram_EventCounter = getHistogram(inputFiles[process], dqmDirectory_afterGenAndRecCuts, "EventCounter");
      if ( !histogram_EventCounter ) continue;
      if ( !(histogram_EventCounter->Integral() > 100.) ) {
	std::cout << process << ": integral(EventCounter) = " << histogram_EventCounter->Integral() << std::endl;
	continue;
      }
      if ( !histograms_plot[iProcess]->GetSumw2N() ) histograms_plot[iProcess]->Sumw2();      
      histograms_plot[iProcess]->Scale(1./histogram_EventCounter->Integral());
      //histograms_plot[iProcess]->Scale(1./histograms_plot[iProcess]->Integral());
      legendEntries_plot[iProcess] = legendEntries[process];
    }

    double legendX0 = 0.50;
    double legendY0 = 0.74;
    if ( plot->meName_.find("recMinusGenLeptonPt") != std::string::npos ||
	 plot->meName_.find("genDeltaPhiLeg1Leg2") != std::string::npos ||
	 plot->meName_.find("genLeg1X")            != std::string::npos ||
	 plot->meName_.find("genLeg2X")            != std::string::npos ) {
      legendX0 = 0.165;
      legendY0 = 0.74;
    }    
    std::string outputFileName = Form("plots/makeEmbeddingValidationPlots_%s_%s.pdf", plot->name_.data(), channel.data());
    showDistribution(800, 900,
		     histograms_plot[0], legendEntries_plot[0],
		     histograms_plot[1], legendEntries_plot[1],
		     histograms_plot[2], legendEntries_plot[2],
		     histograms_plot[3], legendEntries_plot[3],
		     histograms_plot[4], legendEntries_plot[4],
		     histograms_plot[5], legendEntries_plot[5],
		     plot->xMin_, plot->xMax_, plot->numBinsMin_rebinned_, plot->xAxisTitle_, plot->xAxisOffset_,
		     plot->useLogScale_, plot->yMin_, plot->yMax_, plot->yAxisTitle_, plot->yAxisOffset_,
		     legendX0, legendY0,
		     outputFileName);
  }

  std::vector<plotEntryType_efficiency> efficienciesToPlot;
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "electronIdEfficiency_vs_Pt", "Electron Id Efficiency", 
    dqmDirectory_afterGenCuts, "goodElectronEfficiencies/numeratorPt", "goodElectronEfficiencies/denominatorPt", 
    0., 250., 50, "P_{T}^{e} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "electronIdEfficiency_vs_eta", "Electron Id Efficiency", 
    dqmDirectory_afterGenCuts, "goodElectronEfficiencies/numeratorEta", "goodElectronEfficiencies/denominatorEta", 
    -2.5, +2.5, 50, "#eta_{e}", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "electronIsoEfficiency_vs_Pt", "Electron Iso Efficiency", 
    dqmDirectory_afterGenCuts, "goodIsoElectronEfficiencies/numeratorPt", "goodElectronEfficiencies/numeratorPt", 
    0., 250., 50, "P_{T}^{e} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "electronIsoEfficiency_vs_eta", "Electron Iso Efficiency", 
    dqmDirectory_afterGenCuts, "goodIsoElectronEfficiencies/numeratorEta", "goodElectronEfficiencies/numeratorEta", 
    -2.5, +2.5, 50, "#eta_{e}", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "electronIdAndIsoEfficiency_vs_Pt", "Electron Id & Iso Efficiency", 
    dqmDirectory_afterGenCuts, "goodIsoElectronEfficiencies/numeratorPt", "goodIsoElectronEfficiencies/denominatorPt", 
    0., 250., 50, "P_{T}^{e} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "electronIdAndIsoEfficiency_vs_eta", "Electron Id & Iso Efficiency", 
    dqmDirectory_afterGenCuts, "goodIsoElectronEfficiencies/numeratorEta", "goodIsoElectronEfficiencies/denominatorEta", 
    -2.5, +2.5, 50, "#eta_{e}", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "isoElectron12TriggerEfficiency_vs_Pt", "L1_IsoEle12 Efficiency", 
    dqmDirectory_afterGenCuts, "electronTriggerEfficiencyL1_IsoElec12wrtGoodIsoElectrons/numeratorPt", "electronTriggerEfficiencyL1_IsoElec12wrtGoodIsoElectrons/denominatorPt", 
    0., 250., 50, "P_{T}^{e} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "isoElectron12TriggerEfficiency_vs_Eta", "L1_IsoEle12 Efficiency", 
    dqmDirectory_afterGenCuts, "electronTriggerEfficiencyL1_IsoElec12wrtGoodIsoElectrons/numeratorEta", "electronTriggerEfficiencyL1_IsoElec12wrtGoodIsoElectrons/denominatorEta", 
    0., 250., 50, "#eta_{e}", 1.3, 0., 1.4, "#varepsilon", 1.2, false));

/*
  std::map<std::string, std::string> dqmSubDirectories_efficiency;
  dqmSubDirectories_efficiency["gsfElectron"]                    = "genMatchedPatElectronEfficiencies";
  dqmSubDirectories_efficiency["gsfPFElectronBarrelTrkSeed"]     = "genMatchedPatPFElectronBarrelTrkSeedEfficiencies";
  dqmSubDirectories_efficiency["gsfNonPFElectronBarrelTrkSeed"]  = "genMatchedPatNonPFElectronBarrelTrkSeedEfficiencies";
  dqmSubDirectories_efficiency["gsfPFElectronBarrelECALseed"]    = "genMatchedPatPFElectronBarrelECALseedEfficiencies";
  dqmSubDirectories_efficiency["gsfNonPFElectronBarrelECALseed"] = "genMatchedPatNonPFElectronBarrelECALseedEfficiencies";
  dqmSubDirectories_efficiency["gsfPFElectronEndcapTrkSeed"]     = "genMatchedPatPFElectronEndcapTrkSeedEfficiencies";
  dqmSubDirectories_efficiency["gsfNonPFElectronEndcapTrkSeed"]  = "genMatchedPatNonPFElectronEndcapTrkSeedEfficiencies";
  dqmSubDirectories_efficiency["gsfPFElectronEndcapECALseed"]    = "genMatchedPatPFElectronEndcapECALseedEfficiencies";
  dqmSubDirectories_efficiency["gsfNonPFElectronEndcapECALseed"] = "genMatchedPatNonPFElectronEndcapECALseedEfficiencies";
  dqmSubDirectories_efficiency["selElectronMVA"]                 = "selectedElectronIdMVAefficiencies";
  dqmSubDirectories_efficiency["selElectronAntiConv"]            = "selectedElectronConversionVetoEfficiencies";
  for ( std::vector<std::string>::const_iterator electronSelection = electronSelections.begin();
	electronSelection != electronSelections.end(); ++electronSelection ) {
    const std::string& dqmSubDirectory = dqmSubDirectories_efficiency[*electronSelection];
    if ( dqmSubDirectory == "" ) continue;
    efficienciesToPlot.push_back(plotEntryType_efficiency(
      Form("%sIdEfficiency_vs_Pt", electronSelection->data()), "Electron Id Efficiency", 
      dqmDirectory_afterGenCuts, Form("%s/numeratorPt", dqmSubDirectory.data()), Form("%s/denominatorPt", dqmSubDirectory.data()), 
      0., 250., 50, "P_{T}^{e} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
    efficienciesToPlot.push_back(plotEntryType_efficiency(
      Form("%sIdEfficiency_vs_eta", electronSelection->data()), "Electron Id Efficiency",
      dqmDirectory_afterGenCuts, Form("%s/numeratorEta", dqmSubDirectory.data()), Form("%s/denominatorEta", dqmSubDirectory.data()), 
      -2.5, +2.5, 50, "#eta_{e}", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  }
 */
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "muonIdEfficiency_vs_Pt", "Muon Id Efficiency", 
    dqmDirectory_afterGenCuts, "goodMuonEfficiencies/numeratorPt", "goodMuonEfficiencies/denominatorPt", 
    0., 250., 50, "P_{T}^{#mu} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "muonIdEfficiency_vs_eta", "Muon Id Efficiency", 
    dqmDirectory_afterGenCuts, "goodMuonEfficiencies/numeratorEta", "goodMuonEfficiencies/denominatorEta", 
    -2.5, +2.5, 50, "#eta_{#mu}", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "muonIsoEfficiency_vs_Pt", "Muon Iso Efficiency", 
    dqmDirectory_afterGenCuts, "goodIsoMuonEfficiencies/numeratorPt", "goodMuonEfficiencies/numeratorPt", 
    0., 250., 50, "P_{T}^{#mu} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "muonIsoEfficiency_vs_eta", "Muon Iso Efficiency", 
    dqmDirectory_afterGenCuts, "goodIsoMuonEfficiencies/numeratorEta", "goodMuonEfficiencies/numeratorEta", 
    -2.5, +2.5, 50, "#eta_{#mu}", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "muonIdAndIsoEfficiency_vs_Pt", "Muon Id & Iso Efficiency", 
    dqmDirectory_afterGenCuts, "goodIsoMuonEfficiencies/numeratorPt", "goodIsoMuonEfficiencies/denominatorPt", 
    0., 250., 50, "P_{T}^{#mu} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "muonIdAndIsoEfficiency_vs_eta", "Muon Id & Iso Efficiency", 
    dqmDirectory_afterGenCuts, "goodIsoMuonEfficiencies/numeratorEta", "goodIsoMuonEfficiencies/denominatorEta", 
    -2.5, +2.5, 50, "#eta_{#mu}", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "muon8TriggerEfficiency_vs_Pt", "HLT_Mu8 Efficiency", 
    dqmDirectory_afterGenCuts, "muonTriggerEfficiencyL1_Mu8wrtGoodIsoMuons/numeratorPt", "muonTriggerEfficiencyL1_Mu8wrtGoodIsoMuons/denominatorPt", 
    0., 250., 50, "P_{T}^{#mu} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "muon8TriggerEfficiency_vs_eta", "HLT_Mu8 Efficiency", 
    dqmDirectory_afterGenCuts, "muonTriggerEfficiencyL1_Mu8wrtGoodIsoMuons/numeratorEta", "muonTriggerEfficiencyL1_Mu8wrtGoodIsoMuons/denominatorEta", 
    -2.5, +2.5, 50, "#eta_{#mu}", 1.3, 0., 1.4, "#varepsilon", 1.2, false));

  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "tauIdEfficiency_vs_Pt", "Tau Id Efficiency", 
    dqmDirectory_afterGenCuts, "selectedTauEfficiencies/numeratorPt", "selectedTauEfficiencies/denominatorPt", 
    0., 250., 50, "P_{T}^{#tau} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "tauIdEfficiency_vs_eta", "Tau Id Efficiency", 
    dqmDirectory_afterGenCuts, "selectedTauEfficiencies/numeratorEta", "selectedTauEfficiencies/denominatorEta", 
    -2.3, +2.3, 46, "#eta_{#tau}", 1.3, 0., 1.4, "#varepsilon", 1.2, false));

  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "met20TriggerEfficiency", "L1_ETM20 Efficiency", 
    dqmDirectory_afterGenCuts, "metTriggerEfficiencyL1_ETM20_et/numeratorPt", "metTriggerEfficiencyL1_ETM20_et/denominatorPt", 
    0., 250., 50, "E_{T}^{miss} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "met26TriggerEfficiency", "L1_ETM26 Efficiency", 
    dqmDirectory_afterGenCuts, "metTriggerEfficiencyL1_ETM26_et/numeratorPt", "metTriggerEfficiencyL1_ETM26_et/denominatorPt", 
    0., 250., 50, "E_{T}^{miss} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "met30TriggerEfficiency", "L1_ETM30 Efficiency", 
    dqmDirectory_afterGenCuts, "metTriggerEfficiencyL1_ETM30_et/numeratorPt", "metTriggerEfficiencyL1_ETM30_et/denominatorPt", 
    0., 250., 50, "E_{T}^{miss} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "met36TriggerEfficiency", "L1_ETM36 Efficiency", 
    dqmDirectory_afterGenCuts, "metTriggerEfficiencyL1_ETM36_et/numeratorPt", "metTriggerEfficiencyL1_ETM36_et/denominatorPt", 
    0., 250., 50, "E_{T}^{miss} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));
  efficienciesToPlot.push_back(plotEntryType_efficiency(
    "met40TriggerEfficiency", "L1_ETM40 Efficiency", 
    dqmDirectory_afterGenCuts, "metTriggerEfficiencyL1_ETM40_et/numeratorPt", "metTriggerEfficiencyL1_ETM40_et/denominatorPt", 
    0., 250., 50, "E_{T}^{miss} / GeV", 1.3, 0., 1.4, "#varepsilon", 1.2, false));

  for ( std::vector<plotEntryType_efficiency>::const_iterator plot = efficienciesToPlot.begin();
	plot != efficienciesToPlot.end(); ++plot ) {
    std::vector<TH1*> histograms_numerator(6);
    std::vector<TH1*> histograms_denominator(6);
    std::vector<std::string> legendEntries_plot(6);
    unsigned numProcesses = processes.size();
    for ( unsigned iProcess = 0; iProcess < numProcesses; ++iProcess ) {
      const std::string& process = processes[iProcess];
      histograms_numerator[iProcess] = getHistogram(inputFiles[process], "", plot->meName_numerator_);
      histograms_denominator[iProcess] = getHistogram(inputFiles[process], "", plot->meName_denominator_);
      legendEntries_plot[iProcess] = legendEntries[process];
    }

    double legendX0 = 0.50;
    double legendY0 = 0.74;    
    std::string outputFileName = Form("plots/makeEmbeddingValidationPlots_%s_%s.pdf", plot->name_.data(), channel.data());
    showEfficiency(plot->title_, 800, 900,
		   histograms_numerator[0], histograms_denominator[0], legendEntries_plot[0],
		   histograms_numerator[1], histograms_denominator[1], legendEntries_plot[1],
		   histograms_numerator[2], histograms_denominator[2], legendEntries_plot[2],
		   histograms_numerator[3], histograms_denominator[3], legendEntries_plot[3],
		   histograms_numerator[4], histograms_denominator[4], legendEntries_plot[4],
		   histograms_numerator[5], histograms_denominator[5], legendEntries_plot[5],
		   plot->xMin_, plot->xMax_, plot->numBinsMin_rebinned_, plot->xAxisTitle_, plot->xAxisOffset_,
		   plot->yMin_, plot->yMax_, plot->yAxisOffset_,
		   legendX0, legendY0,
		   outputFileName);
  }

  std::vector<std::string> controlPlots;
  controlPlots.push_back(std::string("muonPlusPt"));
  controlPlots.push_back(std::string("muonPlusEta"));
  controlPlots.push_back(std::string("muonMinusPt"));
  controlPlots.push_back(std::string("muonMinusEta"));
  controlPlots.push_back(std::string("diMuonMass"));
  typedef std::pair<int, int> pint;
  std::vector<pint> jetBins;
  jetBins.push_back(pint(-1, -1));
  jetBins.push_back(pint(0, 0));
  jetBins.push_back(pint(1, 1));
  jetBins.push_back(pint(2, 2));
  jetBins.push_back(pint(3, 1000));
  TFile* inputFile = inputFiles["simDYtoMuMu_recEmbedding_angleEq90"];
  for ( std::vector<std::string>::const_iterator controlPlot = controlPlots.begin();
	controlPlot != controlPlots.end(); ++controlPlot ) {
    for ( std::vector<pint>::const_iterator jetBin = jetBins.begin();
	  jetBin != jetBins.end(); ++jetBin ) {

      double xMin, xMax;
      int numBinsX;
      std::string xAxisTitle;
      if ( TString(controlPlot->data()).EndsWith("Pt") ) {
	xMin = 0.;
	xMin = 100.;
	numBinsX = 20;
	xAxisTitle = "P_{T}^{#mu} / GeV";
      } else if ( TString(controlPlot->data()).EndsWith("Eta") ) {
	xMin = -2.5;
	xMin = +2.5;
	numBinsX = 25;
	xAxisTitle = "P_{T}^{#mu} / GeV";
      } else if ( TString(controlPlot->data()).EndsWith("Mass") ) {
	xMin = 0.;
	xMin = 200.;
	numBinsX = 20;
	xAxisTitle = "M_{#mu#mu} / GeV";
      } else assert(0);
      
      int minJets = jetBin->first;
      int maxJets = jetBin->second;
      std::string jetBinLabel;
      if      ( minJets < 0 && maxJets < 0 ) jetBinLabel = "";
      else if (                maxJets < 0 ) jetBinLabel = Form("_numJetsGe%i", minJets);
      else if ( minJets < 0                ) jetBinLabel = Form("_numJetsLe%i", maxJets);
      else if ( maxJets     == minJets     ) jetBinLabel = Form("_numJetsEq%i", minJets);
      else                                   jetBinLabel = Form("_numJets%ito%i", minJets, maxJets);

      std::string histogramName_beforeRad = Form("beforeRad%s/%s_unweighted/", jetBinLabel.data(), controlPlot->data());
      TH1* histogram_beforeRad = getHistogram(inputFile, "validationAnalyzer_mutau", histogramName_beforeRad);
      TH1* histogram_rebinned_beforeRad = rebinHistogram(histogram_beforeRad, numBinsX, xMin, xMax, true);
      TGraphAsymmErrors* graph_beforeRad = convertToGraph(histogram_rebinned_beforeRad);
      
      std::string histogramName_afterRad = Form("afterRad%s/%s_unweighted/", jetBinLabel.data(), controlPlot->data());
      TH1* histogram_afterRad = getHistogram(inputFile, "validationAnalyzer_mutau", histogramName_afterRad);
      TH1* histogram_rebinned_afterRad = rebinHistogram(histogram_afterRad, numBinsX, xMin, xMax, true);
      TGraphAsymmErrors* graph_afterRad = convertToGraph(histogram_rebinned_afterRad);
      
      std::string histogramName_afterRadAndCorr = Form("afterRadAndCorr%s/%s_weighted/", jetBinLabel.data(), controlPlot->data());
      TH1* histogram_afterRadAndCorr = getHistogram(inputFile, "validationAnalyzer_mutau", histogramName_afterRadAndCorr);
      TH1* histogram_rebinned_afterRadAndCorr = rebinHistogram(histogram_afterRadAndCorr, numBinsX, xMin, xMax, true);
      std::string histogramName_afterRadAndCorrErrUp = Form("afterRadAndCorr%s/%s_weightedUp/", jetBinLabel.data(), controlPlot->data());
      TH1* histogram_afterRadAndCorrErrUp = getHistogram(inputFile, "validationAnalyzer_mutau", histogramName_afterRadAndCorrErrUp);
      TH1* histogram_rebinned_afterRadAndCorrErrUp = rebinHistogram(histogram_afterRadAndCorrErrUp, numBinsX, xMin, xMax, true);
      std::string histogramName_afterRadAndCorrErrDown = Form("afterRadAndCorr%s/%s_weightedDown/", jetBinLabel.data(), controlPlot->data());
      TH1* histogram_afterRadAndCorrErrDown = getHistogram(inputFile, "validationAnalyzer_mutau", histogramName_afterRadAndCorrErrDown);
      TH1* histogram_rebinned_afterRadAndCorrErrDown = rebinHistogram(histogram_afterRadAndCorrErrDown, numBinsX, xMin, xMax, true);
      TGraphAsymmErrors* graph_afterRadAndCorr = convertToGraph(histogram_rebinned_afterRadAndCorr, histogram_rebinned_afterRadAndCorrErrUp, histogram_rebinned_afterRadAndCorrErrDown);
      
      double legendX0 = 0.50;
      double legendY0 = 0.74;
      std::string outputFileName = Form("plots/makeEmbeddingValidationPlots_muonRadCorr_%s_%s.pdf", controlPlot->data(), channel.data());
      showGraphs(800, 900,
		 graph_beforeRad, "before Rad.",
		 graph_afterRad, "after Rad.",
		 graph_afterRadAndCorr, "after Rad.+Corr.",
		 0, "",
		 0, "",
		 0, "",
		 xMin, xMax, numBinsX, xAxisTitle, 1.2,
		 true, 0., 1., "a.u.", 1.2, 
		 legendX0, legendY0,
		 outputFileName);
    }
  }

  // CV: close input files
  for ( std::map<std::string, TFile*>::iterator inputFile = inputFiles.begin();
	inputFile != inputFiles.end(); ++inputFile ) {
    delete inputFile->second;
  }
}
