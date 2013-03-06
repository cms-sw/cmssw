
#include <TFile.h>
#include <TString.h>
#include <TH1.h>
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

  //histogram->Scale(1./histogram->Integral());

  return histogram;
}

TH1* rebinHistogram(const std::vector<double>& histogramBinning, const TH1* histogram)
{
  TArrayF histogramBinning_array(histogramBinning.size());
  int idx = 0;
  for ( std::vector<double>::const_iterator binEdge = histogramBinning.begin();
	binEdge != histogramBinning.end(); ++binEdge ) {
    histogramBinning_array[idx] = (*binEdge);
    ++idx;
  }
  std::string histogramName = Form("%s_rebinned", histogram->GetName());
  std::string histogramTitle = histogram->GetTitle();
  int numBins_rebinned = histogramBinning_array.GetSize() - 1;
  //std::cout << "numBins_rebinned = " << numBins_rebinned << std::endl;
  TH1* histogram_rebinned = new TH1D(histogramName.data(), histogramTitle.data(), numBins_rebinned, histogramBinning_array.GetArray());
  TAxis* xAxis = histogram->GetXaxis();
  int numBins = xAxis->GetNbins();
  TAxis* xAxis_rebinned = histogram_rebinned->GetXaxis();
  double binContentSum = 0.;
  double binError2Sum = 0.;  
  int iBin_rebinned = 1;
  double binEdgeLow_rebinned = xAxis_rebinned->GetBinLowEdge(iBin_rebinned);
  for ( int iBin = 1; iBin <= numBins; ++iBin ) {
    double binCenter = xAxis->GetBinCenter(iBin);    
    double binContent = histogram->GetBinContent(iBin);
    double binError = histogram->GetBinError(iBin);    
    bool isNextBin_rebinned = false;
    if ( iBin == numBins ) {
      isNextBin_rebinned = true;
    } else {
      if ( iBin_rebinned < numBins_rebinned && binCenter > xAxis_rebinned->GetBinLowEdge(iBin_rebinned + 1) ) {
	isNextBin_rebinned = true;
      }
    }
    if ( isNextBin_rebinned ) {
      double binWidth_rebinned = xAxis_rebinned->GetBinLowEdge(iBin_rebinned + 1) - binEdgeLow_rebinned;
      histogram_rebinned->SetBinContent(iBin_rebinned, binContentSum/binWidth_rebinned);
      histogram_rebinned->SetBinError(iBin_rebinned, TMath::Sqrt(binError2Sum)/binWidth_rebinned);
      binContentSum = 0.;
      binError2Sum = 0.;
      binEdgeLow_rebinned = xAxis_rebinned->GetBinLowEdge(iBin_rebinned + 1);
      ++iBin_rebinned;
    }
    //std::cout << "binCenter = " << binCenter << ": iBin = " << iBin << ", iBin_rebinned = " << iBin_rebinned << std::endl;
    binContentSum += binContent;
    binError2Sum += (binError*binError);    
  }
  return histogram_rebinned;
}

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

  return histogramRatio;
}

void showDistribution(double canvasSizeX, double canvasSizeY,
		      TH1* histogram_ref, const std::string& legendEntry_ref,
		      TH1* histogram2, const std::string& legendEntry2,
		      TH1* histogram3, const std::string& legendEntry3,
		      TH1* histogram4, const std::string& legendEntry4,
		      TH1* histogram5, const std::string& legendEntry5,
		      TH1* histogram6, const std::string& legendEntry6,
		      const std::string& xAxisTitle, double xAxisOffset,
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

  int colors[6] = { 1, 2, 3, 4, 6, 7 };
  int markerStyles[6] = { 22, 32, 20, 24, 21, 25 };

  TLegend* legend = new TLegend(legendX0, legendY0, legendX0 + 0.44, legendY0 + 0.20, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);

  histogram_ref->SetTitle("");
  histogram_ref->SetStats(false);
  histogram_ref->SetMinimum(yMin);
  histogram_ref->SetMaximum(yMax);
  histogram_ref->SetLineColor(colors[0]);
  histogram_ref->SetLineWidth(2);
  histogram_ref->SetMarkerColor(colors[0]);
  histogram_ref->SetMarkerStyle(markerStyles[0]);
  histogram_ref->Draw("e1p");
  legend->AddEntry(histogram_ref, legendEntry_ref.data(), "p");

  TAxis* xAxis_top = histogram_ref->GetXaxis();
  xAxis_top->SetTitle(xAxisTitle.data());
  xAxis_top->SetTitleOffset(xAxisOffset);
  xAxis_top->SetLabelColor(10);
  xAxis_top->SetTitleColor(10);

  TAxis* yAxis_top = histogram_ref->GetYaxis();
  yAxis_top->SetTitle(yAxisTitle.data());
  yAxis_top->SetTitleOffset(yAxisOffset);

  if ( histogram2 ) {
    histogram2->SetLineColor(colors[1]);
    histogram2->SetLineWidth(2);
    histogram2->SetMarkerColor(colors[1]);
    histogram2->SetMarkerStyle(markerStyles[1]);
    histogram2->Draw("e1psame");
    legend->AddEntry(histogram2, legendEntry2.data(), "p");
  }

  if ( histogram3 ) {
    histogram3->SetLineColor(colors[2]);
    histogram3->SetLineWidth(2);
    histogram3->SetMarkerColor(colors[2]);
    histogram3->SetMarkerStyle(markerStyles[2]);
    histogram3->Draw("e1psame");
    legend->AddEntry(histogram3, legendEntry3.data(), "p");
  }

  if ( histogram4 ) {
    histogram4->SetLineColor(colors[3]);
    histogram4->SetLineWidth(2);
    histogram4->SetMarkerColor(colors[3]);
    histogram4->SetMarkerStyle(markerStyles[3]);
    histogram4->Draw("e1psame");
    legend->AddEntry(histogram4, legendEntry4.data(), "p");
  }

  if ( histogram5 ) {
    histogram5->SetLineColor(colors[4]);
    histogram5->SetLineWidth(2);
    histogram5->SetMarkerColor(colors[4]);
    histogram5->SetMarkerStyle(markerStyles[4]);
    histogram5->Draw("e1psame");
    legend->AddEntry(histogram5, legendEntry5.data(), "p");
  }

  if ( histogram6 ) {
    histogram6->SetLineColor(colors[5]);
    histogram6->SetLineWidth(2);
    histogram6->SetMarkerColor(colors[5]);
    histogram6->SetMarkerStyle(markerStyles[5]);
    histogram6->Draw("e1psame");
    legend->AddEntry(histogram6, legendEntry6.data(), "p");
  }

  legend->Draw();

  canvas->cd();
  bottomPad->Draw();
  bottomPad->cd();

  TH1* histogram2_div_ref = 0;
  if ( histogram2 ) {
    std::string histogramName2_div_ref = std::string(histogram2->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram2_div_ref = compRatioHistogram(histogramName2_div_ref, histogram2, histogram_ref);
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
    histogram3_div_ref = compRatioHistogram(histogramName3_div_ref, histogram3, histogram_ref);
    histogram3_div_ref->Draw("e1psame");
  }

  TH1* histogram4_div_ref = 0;
  if ( histogram4 ) {
    std::string histogramName4_div_ref = std::string(histogram4->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram4_div_ref = compRatioHistogram(histogramName4_div_ref, histogram4, histogram_ref);
    histogram4_div_ref->Draw("e1psame");
  }

  TH1* histogram5_div_ref = 0;
  if ( histogram5 ) {
    std::string histogramName5_div_ref = std::string(histogram5->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram5_div_ref = compRatioHistogram(histogramName5_div_ref, histogram5, histogram_ref);
    histogram5_div_ref->Draw("e1psame");
  }

  TH1* histogram6_div_ref = 0;
  if ( histogram6 ) {
    std::string histogramName6_div_ref = std::string(histogram6->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram6_div_ref = compRatioHistogram(histogramName6_div_ref, histogram6, histogram_ref);
    histogram6_div_ref->Draw("e1psame");
  }
  
  canvas->Update();
  size_t idx = outputFileName.find_last_of('.');
  std::string outputFileName_plot = std::string(outputFileName, 0, idx);
  if ( useLogScale ) outputFileName_plot.append("_log");
  else outputFileName_plot.append("_linear");
  if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
  canvas->Print(std::string(outputFileName_plot).append(".png").data());
  //canvas->Print(std::string(outputFileName_plot).append(".pdf").data());
  
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

void makeEmbeddingReweightLUTs2()
{
//--- stop ROOT from keeping references to all histograms
  TH1::AddDirectory(false);

//--- suppress the output canvas 
  gROOT->SetBatch(true);

  std::string inputFilePath = "/data1/veelken/tmp/EmbeddingValidation/";

  std::string inputFileName_numerator   = "validateMCEmbedding_simDYtoTauTau_all_v1_9_2.root";
  std::string inputFileName_denominator = "validateMCEmbedding_simDYtoMuMu_noEvtSel_embedEqRH_cleanEqDEDX_replaceGenMuons_by_mutau_embedAngleEq90_muonCaloSF1_0_all_v1_9_2.root";

  TFile* inputFile_numerator = new TFile(std::string(inputFilePath).append(inputFileName_numerator).data(), "READ");
  TFile* inputFile_denominator = new TFile(std::string(inputFilePath).append(inputFileName_denominator).data(), "READ");

  std::vector<double> genDiTauMassBinning;
  double genDiTauMass = 0.;
std::cout << "break-point 1 reached" << std::endl;
  while ( genDiTauMass <= 250. ) {
    genDiTauMassBinning.push_back(genDiTauMass);
    if      ( genDiTauMass <  50. ) genDiTauMass += 50.;
    else if ( genDiTauMass <  75. ) genDiTauMass +=  5.;
    else if ( genDiTauMass < 105. ) genDiTauMass +=  1.;
    else if ( genDiTauMass < 130. ) genDiTauMass +=  5.;
    else if ( genDiTauMass < 150. ) genDiTauMass += 10.;
    else if ( genDiTauMass < 200. ) genDiTauMass += 25.;
    else                            genDiTauMass += 50.;
std::cout << "break-point 2 reached" << std::endl;
std::cout << genDiTauMass << std::endl;
  }
std::cout << "break-point 3 reached" << std::endl;
  TH1* histogramDiTauMass_numerator = getHistogram(inputFile_numerator, "validationAnalyzer_mutau", "genDiTauMass");
  TH1* histogramDiTauMass_numerator_rebinned = rebinHistogram(genDiTauMassBinning, histogramDiTauMass_numerator);
  TH1* histogramDiTauMass_denominator = getHistogram(inputFile_numerator, "validationAnalyzer_mutau", "genDiTauMass");
  TH1* histogramDiTauMass_denominator_rebinned = rebinHistogram(genDiTauMassBinning, histogramDiTauMass_denominator);
  TH1* histogramDiTauMass_ratio = compRatioHistogram("embeddingReweight_genDiTauMass", histogramDiTauMass_numerator_rebinned, histogramDiTauMass_denominator_rebinned);
/*
  showDistribution(800, 900,
		   histogramDiTauMass_numerator_rebinned, "Embedding",
		   histogramDiTauMass_denominator_rebinned, "gen. Z/#gamma^{*} #rightarrow #tau #tau",
		   0, "",
		   0, "",
		   0, "",
		   0, "",
		   "M_{#tau#tau}", 1.3,
		   true, 1.e-6, 1.e+0, "a.u.", 1.2, 
		   0.50, 0.74,
		   "makeEmbeddingReweightLUTs_genDiTauMass.png");
*/
std::cout << "break-point 4 reached" << std::endl;
  std::vector<double> genDiTauPtBinning;
  double genDiTauPt = 0.;
  while ( genDiTauPt <= 250. ) {
    genDiTauPtBinning.push_back(genDiTauPt);
    if      ( genDiTauPt <  25. ) genDiTauPt +=  1.;
    else if ( genDiTauPt <  50. ) genDiTauPt +=  2.5;
    else if ( genDiTauPt < 100. ) genDiTauPt +=  5.;
    else if ( genDiTauPt < 150. ) genDiTauPt += 10.;
    else if ( genDiTauPt < 200. ) genDiTauPt += 25.;
    else                          genDiTauPt += 50.;
std::cout << "break-point 5 reached" << std::endl;
  }
std::cout << "break-point 6 reached" << std::endl;
  TH1* histogramDiTauPt_numerator = getHistogram(inputFile_numerator, "validationAnalyzer_mutau", "genDiTauPt");
  TH1* histogramDiTauPt_numerator_rebinned = rebinHistogram(genDiTauPtBinning, histogramDiTauPt_numerator);
  TH1* histogramDiTauPt_denominator = getHistogram(inputFile_numerator, "validationAnalyzer_mutau", "genDiTauPt");
  TH1* histogramDiTauPt_denominator_rebinned = rebinHistogram(genDiTauPtBinning, histogramDiTauPt_denominator);
  TH1* histogramDiTauPt_ratio = compRatioHistogram("embeddingReweight_genDiTauPt", histogramDiTauPt_numerator_rebinned, histogramDiTauPt_denominator_rebinned);
  /*
  showDistribution(800, 900,
		   histogramDiTauPt_numerator_rebinned, "Embedding",
		   histogramDiTauPt_denominator_rebinned, "gen. Z/#gamma^{*} #rightarrow #tau #tau",
		   0, "",
		   0, "",
		   0, "",
		   0, "",
		   "P_{T}^{#tau#tau}", 1.3,
		   true, 1.e-6, 1.e+0, "a.u.", 1.2, 
		   0.50, 0.74,
		   "makeEmbeddingReweightLUTs_genDiTauPt.png");
  TF1* fitDiTauPt = new TF1("fitDiTauPt", "([0] + [1]*x)*(1.0 + [2]*TMath::TanH([3] + [4]*x))", 0., 250.);
  histogramDiTauPt_ratio->Fit(fitDiTauPt);
*/
std::cout << "break-point 7 reached" << std::endl;
  TFile* outputFile = new TFile("makeEmbeddingReweightLUTs.root", "RECREATE");
  histogramDiTauMass_ratio->Write();
  histogramDiTauPt_ratio->Write();
  delete outputFile;

  delete inputFile_numerator;
  delete inputFile_denominator;

  std::cout << "BLAH!!" << std::endl;
}
