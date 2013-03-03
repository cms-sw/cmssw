
#include <TFile.h>
#include <TString.h>
#include <TH1.h>
#include <TH2.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TTree.h>
#include <TLorentzVector.h>
#include <TMath.h>
#include <TROOT.h>

#include <vector>
#include <iostream>
#include <iomanip>

enum { kUndefined, kGenTau1Pt, kGenTau2Pt, kGenDiTauPt, kGenDiTauMass };

struct weightEntryType
{
  weightEntryType(const TString& name, const TH1* lut, int variableX)
    : name_(name),
      lut_(lut),
      xAxis_(lut->GetXaxis()),
      numBinsX_(lut->GetNbinsX()),
      yAxis_(0),
      numBinsY_(-1),
      variableX_(variableX),            
      variableY_(kUndefined)
  {}
  weightEntryType(const TString& name, const TH2* lut, int variableX, int variableY)
    : name_(name),
      lut_(lut),
      xAxis_(lut->GetXaxis()),
      numBinsX_(lut->GetNbinsX()),
      yAxis_(lut->GetYaxis()),
      numBinsY_(lut->GetNbinsY()),
      variableX_(variableX),            
      variableY_(variableY)
  {}
  Float_t operator()(Float_t genTau1Pt, Float_t genTau2Pt, Float_t genDiTauPt, Float_t genDiTauMass)
  {
    //std::cout << "<weightEntryType::operator()>:" << std::endl;
    //std::cout << " name = " << name_ << std::endl;
    Float_t weight = 1.0;
    if ( yAxis_ ) { // 2d case
      Float_t variableX_value = getVariableValue(genTau1Pt, genTau2Pt, genDiTauPt, genDiTauMass, variableX_);
      int binX = xAxis_->FindBin(variableX_value);
      if ( binX <= 1 ) binX = 1;
      else if ( binX >= numBinsX_ ) binX = numBinsX_;
      Float_t variableY_value = getVariableValue(genTau1Pt, genTau2Pt, genDiTauPt, genDiTauMass, variableY_);
      int binY = yAxis_->FindBin(variableY_value);
      if ( binY <= 1 ) binY = 1;
      else if ( binY >= numBinsY_ ) binY = numBinsY_;
      weight = lut_->GetBinContent(binX, binY);
    } else {        // 1d case
      Float_t variableX_value = getVariableValue(genTau1Pt, genTau2Pt, genDiTauPt, genDiTauMass, variableX_);
      int binX = xAxis_->FindBin(variableX_value);
      if ( binX <= 1 ) binX = 1;
      else if ( binX >= numBinsX_ ) binX = numBinsX_;
      weight = lut_->GetBinContent(binX);
    }
    //std::cout << "--> weight = " << weight << std::endl;
    return weight;
  }
  Float_t getVariableValue(Float_t genTau1Pt, Float_t genTau2Pt, Float_t genDiTauPt, Float_t genDiTauMass, int variable)
  {
    if      ( variable == kGenTau1Pt    ) return genTau1Pt;
    else if ( variable == kGenTau2Pt    ) return genTau2Pt;
    else if ( variable == kGenDiTauPt   ) return genDiTauPt;
    else if ( variable == kGenDiTauMass ) return genDiTauMass;
    else assert(0);
  }
  TString name_;
  const TH1* lut_;
  TAxis* xAxis_;
  Int_t numBinsX_;
  TAxis* yAxis_;
  Int_t numBinsY_;
  int variableX_;
  int variableY_;
};

struct plotEntryType
{
  plotEntryType(const TString& name, Bool_t isEmbedded)
    : name_(name),
      isEmbedded_(isEmbedded),
      histogramGenTau1Pt_(0),
      histogramGenTau1Eta_(0),
      histogramGenTau2Pt_(0),
      histogramGenTau2Eta_(0),
      histogramGenDiTauPt_(0),
      histogramGenDiTauMass_(0)
  {    
    histogramGenTau1Pt_ = bookHistogram1d("genTau1Pt", name, 50, 0., 250.);
    histogramGenTau1Eta_ = bookHistogram1d("genTau1Eta", name, 50, -2.5, +2.5);
    histogramGenTau2Pt_ = bookHistogram1d("genTau2Pt", name, 50, 0., 250.);
    histogramGenTau2Eta_ = bookHistogram1d("genTau2Eta", name, 50, -2.5, +2.5);
    std::vector<double> genTauPtBinning;
    double genTauPt = 0.;
    while ( genTauPt <= 250. ) {
      genTauPtBinning.push_back(genTauPt);
      if      ( genTauPt <  20. ) genTauPt +=  5.;
      else if ( genTauPt <  40. ) genTauPt +=  2.;
      else if ( genTauPt <  50. ) genTauPt +=  5.;
      else if ( genTauPt <  80. ) genTauPt += 10.;
      else if ( genTauPt < 100. ) genTauPt += 20.;
      else if ( genTauPt < 150. ) genTauPt += 50.;
      else                          genTauPt += 100.;
    }
    histogramGenTau2Pt_vs_GenTau1Pt_ = bookHistogram2d("genTau2Pt_vs_genTau1Pt", name, genTauPtBinning, genTauPtBinning);
    std::vector<double> genDiTauPtBinning;
    double genDiTauPt = 0.;
    while ( genDiTauPt <= 250. ) {
      genDiTauPtBinning.push_back(genDiTauPt);
      if      ( genDiTauPt <  20. ) genDiTauPt +=   2.;
      else if ( genDiTauPt <  40. ) genDiTauPt +=   5.;
      else if ( genDiTauPt <  50. ) genDiTauPt +=  10.;
      else if ( genDiTauPt <  75. ) genDiTauPt +=  25.;
      else if ( genDiTauPt < 150. ) genDiTauPt +=  75.;
      else                          genDiTauPt += 100.;
    }
    histogramGenDiTauPt_ = bookHistogram1d("genDiTauPt", name, 50, 0., 250.);   
    std::vector<double> genDiTauMassBinning;
    double genDiTauMass = 0.;
    while ( genDiTauMass <= 250. ) {
      genDiTauMassBinning.push_back(genDiTauMass);
      if      ( genDiTauMass <  50. ) genDiTauMass +=  50.;
      else if ( genDiTauMass <  75. ) genDiTauMass +=   5.;
      else if ( genDiTauMass < 105. ) genDiTauMass +=   1.;
      else if ( genDiTauMass < 130. ) genDiTauMass +=   5.;
      else if ( genDiTauMass < 150. ) genDiTauMass +=  20.;
      else                            genDiTauMass += 100.;
    }
    histogramGenDiTauMass_ = bookHistogram1d("genDiTauMass", name, 50, 0., 250.);
  }
  ~plotEntryType()
  {
    delete histogramGenTau1Pt_;
    delete histogramGenTau1Eta_;
    delete histogramGenTau2Pt_;
    delete histogramGenTau2Eta_;
    delete histogramGenTau2Pt_vs_GenTau1Pt_;
    delete histogramGenDiTauPt_;
    delete histogramGenDiTauMass_;
  }
  TH1* bookHistogram1d(const TString& name1, const TString& name2, Int_t numBinsX, Float_t xMin, Float_t xMax)
  {
    TString histogramName = Form("%s_%s", name1.Data(), name2.Data());
    TH1* histogram = new TH1D(histogramName.Data(), histogramName.Data(), numBinsX, xMin, xMax);
    return histogram;
  }
  TH1* bookHistogram1d(const TString& name1, const TString& name2, const std::vector<double>& binningX)
  {
    TString histogramName = Form("%s_%s", name1.Data(), name2.Data());
    TArrayF binningX_array(binningX.size());
    for ( size_t iBinX = 0; iBinX < binningX.size(); ++iBinX ) {
      binningX_array[iBinX] = binningX[iBinX];
    }
    Int_t numBinsX = binningX_array.GetSize() - 1;
    TH1* histogram = new TH1D(histogramName.Data(), histogramName.Data(), numBinsX, binningX_array.GetArray());
    return histogram;
  }
  TH2* bookHistogram2d(const TString& name1, const TString& name2, Int_t numBinsX, Float_t xMin, Float_t xMax, Int_t numBinsY, Float_t yMin, Float_t yMax)
  {
    TString histogramName = Form("%s_%s", name1.Data(), name2.Data());
    TH2* histogram = new TH2D(histogramName.Data(), histogramName.Data(), numBinsX, xMin, xMax, numBinsY, yMin, yMax);
    return histogram;
  }
  TH2* bookHistogram2d(const TString& name1, const TString& name2, const std::vector<double>& binningX, const std::vector<double>& binningY)
  {
    TString histogramName = Form("%s_%s", name1.Data(), name2.Data());
    TArrayF binningX_array(binningX.size());
    for ( size_t iBinX = 0; iBinX < binningX.size(); ++iBinX ) {
      binningX_array[iBinX] = binningX[iBinX];
    }
    Int_t numBinsX = binningX_array.GetSize() - 1;
    TArrayF binningY_array(binningY.size());
    for ( size_t iBinY = 0; iBinY < binningY.size(); ++iBinY ) {
      binningY_array[iBinY] = binningY[iBinY];
    }
    Int_t numBinsY = binningY_array.GetSize() - 1;
    TH2* histogram = new TH2D(histogramName.Data(), histogramName.Data(), numBinsX, binningX_array.GetArray(), numBinsY, binningY_array.GetArray());
    return histogram;
  }
  void fillHistograms(TTree* tree, const std::vector<weightEntryType*>& weightEntries, int maxEvents)
  {
    Float_t genDiTauEn, genDiTauPx, genDiTauPy, genDiTauPz;
    Float_t genTau1En, genTau1Px, genTau1Py, genTau1Pz;
    Float_t genTau2En, genTau2Px, genTau2Py, genTau2Pz;
    
    Float_t muonRadCorrWeight;
    Float_t TauSpinnerWeight;
    Float_t genFilterWeight;

    tree->SetBranchAddress("genDiTauEn", &genDiTauEn);
    tree->SetBranchAddress("genDiTauPx", &genDiTauPx);
    tree->SetBranchAddress("genDiTauPy", &genDiTauPy);
    tree->SetBranchAddress("genDiTauPz", &genDiTauPz);
    tree->SetBranchAddress("genTau1En", &genTau1En);
    tree->SetBranchAddress("genTau1Px", &genTau1Px);
    tree->SetBranchAddress("genTau1Py", &genTau1Py);
    tree->SetBranchAddress("genTau1Pz", &genTau1Pz);
    tree->SetBranchAddress("genTau2En", &genTau2En);
    tree->SetBranchAddress("genTau2Px", &genTau2Px);
    tree->SetBranchAddress("genTau2Py", &genTau2Py);
    tree->SetBranchAddress("genTau2Pz", &genTau2Pz);

    if ( isEmbedded_ ) {
      tree->SetBranchAddress("muonRadiationCorrWeightProducer_weight", &muonRadCorrWeight);
      tree->SetBranchAddress("TauSpinnerReco_TauSpinnerWT", &TauSpinnerWeight);
      tree->SetBranchAddress("genFilterInfo", &genFilterWeight);
    }

    int numEntries = tree->GetEntries();
    for ( int iEntry = 0; iEntry < numEntries && (iEntry < maxEvents || maxEvents == -1); ++iEntry ) {
      if ( iEntry > 0 && (iEntry % 10000) == 0 ) {
	std::cout << "processing Event " << iEntry << std::endl;
      }
      
      tree->GetEntry(iEntry);

      TLorentzVector genDiTauP4(genDiTauPx, genDiTauPy, genDiTauPz, genDiTauEn);
      TLorentzVector genTau1P4(genTau1Px, genTau1Py, genTau1Pz, genTau1En);
      TLorentzVector genTau2P4(genTau2Px, genTau2Py, genTau2Pz, genTau2En);

      Float_t evtWeight = 1.0;
      if ( isEmbedded_ ) {
	evtWeight *= muonRadCorrWeight;
	evtWeight *= TauSpinnerWeight;
	evtWeight *= genFilterWeight;
      }
      for ( std::vector<weightEntryType*>::const_iterator weightEntry = weightEntries.begin();
	    weightEntry != weightEntries.end(); ++weightEntry ) {
	evtWeight *= (**weightEntry)(genTau1P4.Pt(), genTau2P4.Pt(), genDiTauP4.Pt(), genDiTauP4.M());
      }

      if ( evtWeight < 1.e-3 || evtWeight > 1.e+3 || TMath::IsNaN(evtWeight) ) {
	std::cout << "muonRadCorrWeight = " << muonRadCorrWeight << std::endl;
	std::cout << "TauSpinnerWeight = " << TauSpinnerWeight << std::endl;
	std::cout << "genFilterWeight = " << genFilterWeight << std::endl;
	for ( std::vector<weightEntryType*>::const_iterator weightEntry = weightEntries.begin();
	      weightEntry != weightEntries.end(); ++weightEntry ) {
	  std::cout << (*weightEntry)->name_ << " = " << (**weightEntry)(genTau1P4.Pt(), genTau2P4.Pt(), genDiTauP4.Pt(), genDiTauP4.M()) << std::endl;
	}
	continue;
      }

      histogramGenTau1Pt_->Fill(genTau1P4.Pt(), evtWeight);
      histogramGenTau1Eta_->Fill(genTau1P4.Eta(), evtWeight);
      histogramGenTau2Pt_->Fill(genTau2P4.Pt(), evtWeight);
      histogramGenTau2Eta_->Fill(genTau2P4.Eta(), evtWeight);
      histogramGenTau2Pt_vs_GenTau1Pt_->Fill(genTau1P4.Pt(), genTau2P4.Pt(), evtWeight);
      histogramGenDiTauPt_->Fill(genDiTauP4.Pt(), evtWeight);
      histogramGenDiTauMass_->Fill(genDiTauP4.M(), evtWeight);
    }
  }
  void saveHistograms()
  {
    histogramGenTau1Pt_->Write();
    histogramGenTau1Eta_->Write();
    histogramGenTau2Pt_->Write();
    histogramGenTau2Eta_->Write();
    histogramGenTau2Pt_vs_GenTau1Pt_->Write();
    histogramGenDiTauPt_->Write();
    histogramGenDiTauMass_->Write();
  }
  TString name_;
  Bool_t isEmbedded_;
  TH1* histogramGenTau1Pt_;
  TH1* histogramGenTau1Eta_;
  TH1* histogramGenTau2Pt_;
  TH1* histogramGenTau2Eta_;
  TH2* histogramGenTau2Pt_vs_GenTau1Pt_;
  TH1* histogramGenDiTauPt_;
  TH1* histogramGenDiTauMass_;
};

//-------------------------------------------------------------------------------
TH1* compRatioHistogram(const std::string& ratioHistogramName, const TH1* numerator, const TH1* denominator, Float_t offset = 0.)
{
  //std::cout << "<compRatioHistogram>:" << std::endl;
  //std::cout << " numerator: name = " << numerator->GetName() << ", integral = " << numerator->Integral() << std::endl;
  //std::cout << " denominator: name = " << denominator->GetName() << ", integral = " << denominator->Integral() << std::endl;

  assert(numerator->GetDimension() == denominator->GetDimension());
  assert(numerator->GetNbinsX() == denominator->GetNbinsX());
  assert(numerator->GetNbinsY() == denominator->GetNbinsY());

  TH1* histogramRatio = (TH1*)numerator->Clone(ratioHistogramName.data());
  histogramRatio->Divide(denominator);
  
  if ( offset != 0. ) {    
    if ( numerator->GetDimension() == 1 ) {
      int numBinsX = histogramRatio->GetNbinsX();
      for ( int iBinX = 1; iBinX <= numBinsX; ++iBinX ){
	double binContent = histogramRatio->GetBinContent(iBinX);
	histogramRatio->SetBinContent(iBinX, binContent - offset);
      }
    } else if ( numerator->GetDimension() == 2 ) {
      int numBinsX = histogramRatio->GetNbinsX();
      int numBinsY = histogramRatio->GetNbinsY();
      for ( int iBinX = 1; iBinX <= numBinsX; ++iBinX ){
	for ( int iBinY = 1; iBinY <= numBinsY; ++iBinY ){
	  double binContent = histogramRatio->GetBinContent(iBinX, iBinY);
	  histogramRatio->SetBinContent(iBinX, iBinY, binContent - offset);
	}
      }
    } else assert(0);
  }

  histogramRatio->SetLineColor(numerator->GetLineColor());
  histogramRatio->SetLineWidth(numerator->GetLineWidth());
  histogramRatio->SetMarkerColor(numerator->GetMarkerColor());
  histogramRatio->SetMarkerStyle(numerator->GetMarkerStyle());

  return histogramRatio;
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

  int colors[6] = { 1, 2, 3, 4, 6, 7 };
  int markerStyles[6] = { 22, 32, 20, 24, 21, 25 };

  TLegend* legend = new TLegend(legendX0, legendY0, 0.94, 0.94, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);

  TH1* histogram_ref_rebinned = rebinHistogram(histogram_ref, numBinsMin_rebinned, xMin, xMax, true);
  histogram_ref_rebinned->SetTitle("");
  histogram_ref_rebinned->SetStats(false);
  histogram_ref_rebinned->SetMinimum(yMin);
  histogram_ref_rebinned->SetMaximum(yMax);
  histogram_ref_rebinned->SetLineColor(colors[0]);
  histogram_ref_rebinned->SetLineWidth(2);
  histogram_ref_rebinned->SetMarkerColor(colors[0]);
  histogram_ref_rebinned->SetMarkerStyle(markerStyles[0]);
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
    histogram2_rebinned = rebinHistogram(histogram2, numBinsMin_rebinned, xMin, xMax, true);
    histogram2_rebinned->SetLineColor(colors[1]);
    histogram2_rebinned->SetLineWidth(2);
    histogram2_rebinned->SetMarkerColor(colors[1]);
    histogram2_rebinned->SetMarkerStyle(markerStyles[1]);
    histogram2_rebinned->Draw("e1psame");
    legend->AddEntry(histogram2_rebinned, legendEntry2.data(), "p");
  }

  TH1* histogram3_rebinned = 0;
  if ( histogram3 ) {
    histogram3_rebinned = rebinHistogram(histogram3, numBinsMin_rebinned, xMin, xMax, true);
    histogram3_rebinned->SetLineColor(colors[2]);
    histogram3_rebinned->SetLineWidth(2);
    histogram3_rebinned->SetMarkerColor(colors[2]);
    histogram3_rebinned->SetMarkerStyle(markerStyles[2]);
    histogram3_rebinned->Draw("e1psame");
    legend->AddEntry(histogram3_rebinned, legendEntry3.data(), "p");
  }

  TH1* histogram4_rebinned = 0;
  if ( histogram4 ) {
    histogram4_rebinned = rebinHistogram(histogram4, numBinsMin_rebinned, xMin, xMax, true);
    histogram4_rebinned->SetLineColor(colors[3]);
    histogram4_rebinned->SetLineWidth(2);
    histogram4_rebinned->SetMarkerColor(colors[3]);
    histogram4_rebinned->SetMarkerStyle(markerStyles[3]);
    histogram4_rebinned->Draw("e1psame");
    legend->AddEntry(histogram4_rebinned, legendEntry4.data(), "p");
  }

  TH1* histogram5_rebinned = 0;
  if ( histogram5 ) {
    histogram5_rebinned = rebinHistogram(histogram5, numBinsMin_rebinned, xMin, xMax, true);
    histogram5_rebinned->SetLineColor(colors[4]);
    histogram5_rebinned->SetLineWidth(2);
    histogram5_rebinned->SetMarkerColor(colors[4]);
    histogram5_rebinned->SetMarkerStyle(markerStyles[4]);
    histogram5_rebinned->Draw("e1psame");
    legend->AddEntry(histogram5_rebinned, legendEntry5.data(), "p");
  }

  TH1* histogram6_rebinned = 0;
  if ( histogram6 ) {
    histogram6_rebinned = rebinHistogram(histogram6, numBinsMin_rebinned, xMin, xMax, true);
    histogram6_rebinned->SetLineColor(colors[5]);
    histogram6_rebinned->SetLineWidth(2);
    histogram6_rebinned->SetMarkerColor(colors[5]);
    histogram6_rebinned->SetMarkerStyle(markerStyles[5]);
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
    histogram2_div_ref = compRatioHistogram(histogramName2_div_ref, histogram2_rebinned, histogram_ref_rebinned, 1.0);
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
    histogram3_div_ref = compRatioHistogram(histogramName3_div_ref, histogram3_rebinned, histogram_ref_rebinned, 1.0);
    histogram3_div_ref->Draw("e1psame");
  }

  TH1* histogram4_div_ref = 0;
  if ( histogram4 ) {
    std::string histogramName4_div_ref = std::string(histogram4->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram4_div_ref = compRatioHistogram(histogramName4_div_ref, histogram4_rebinned, histogram_ref_rebinned, 1.0);
    histogram4_div_ref->Draw("e1psame");
  }

  TH1* histogram5_div_ref = 0;
  if ( histogram5 ) {
    std::string histogramName5_div_ref = std::string(histogram5->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram5_div_ref = compRatioHistogram(histogramName5_div_ref, histogram5_rebinned, histogram_ref_rebinned, 1.0);
    histogram5_div_ref->Draw("e1psame");
  }

  TH1* histogram6_div_ref = 0;
  if ( histogram6 ) {
    std::string histogramName6_div_ref = std::string(histogram6->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram6_div_ref = compRatioHistogram(histogramName6_div_ref, histogram6_rebinned, histogram_ref_rebinned, 1.0);
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

void makeEmbeddingKineReweightLUTs2()
{
  gROOT->SetBatch(true);

  TString inputFileName_Ztautau  = "/data1/veelken/tmp/EmbeddingValidation/embeddingKineReweightNtuple_simDYtoTauTau_mutau_v1_9_9_all.root";
  TString inputFileName_Embedded = "/data1/veelken/tmp/EmbeddingValidation/embeddingKineReweightNtuple_simDYtoMuMu_embedEqRH_replaceGenMuons_by_mutau_v1_9_9_all.root";

  TString treeName = "embeddingKineReweightNtupleProducer/embeddingKineReweightNtuple";

  TFile* inputFile_Ztautau = TFile::Open(inputFileName_Ztautau.Data());
  if ( !inputFile_Ztautau ) {
    std::cerr << "Failed to open input file = " << inputFileName_Ztautau.Data() << " !!" << std::endl;
    assert(0);
  }
  TTree* tree_Ztautau = dynamic_cast<TTree*>(inputFile_Ztautau->Get(treeName.Data()));
  if ( !tree_Ztautau ) {
    std::cerr << "Failed to find tree = " << treeName.Data() << " in input file = " << inputFileName_Ztautau.Data() << " !!" << std::endl;
    assert(0);
  }
  std::cout << "tree_Ztautau has " << tree_Ztautau->GetEntries() << " entries." << std::endl;

  TFile* inputFile_Embedded = TFile::Open(inputFileName_Embedded.Data());
  if ( !inputFile_Embedded ) {
    std::cerr << "Failed to open input file = " << inputFileName_Embedded.Data() << " !!" << std::endl;
    assert(0);
  }
  TTree* tree_Embedded = dynamic_cast<TTree*>(inputFile_Embedded->Get(treeName.Data()));
  if ( !tree_Embedded ) {
    std::cerr << "Failed to find tree = " << treeName.Data() << " in input file = " << inputFileName_Embedded.Data() << " !!" << std::endl;
    assert(0);
  }
  std::cout << "tree_Embedded has " << tree_Embedded->GetEntries() << " entries." << std::endl;

  int maxEvents = -1;
  
  plotEntryType* plots_Ztautau = new plotEntryType("Ztautau", false);
  plots_Ztautau->fillHistograms(tree_Ztautau, std::vector<weightEntryType*>(), maxEvents);
  
  plotEntryType* plots_Embedded = new plotEntryType("Embedded", true);
  plots_Embedded->fillHistograms(tree_Embedded, std::vector<weightEntryType*>(), maxEvents);

  plotEntryType* plots_Embedded_reweighted1 = new plotEntryType("Embedded_reweighted1", true);
  TH1* lut_reweight1 = compRatioHistogram("lut_reweight1", plots_Ztautau->histogramGenDiTauMass_, plots_Embedded->histogramGenDiTauMass_);
  std::vector<weightEntryType*> weightEntries_Embedded_reweighted1;
  weightEntries_Embedded_reweighted1.push_back(new weightEntryType("Embedded_reweighted1", lut_reweight1, kGenDiTauMass));
  plots_Embedded_reweighted1->fillHistograms(tree_Embedded, weightEntries_Embedded_reweighted1, maxEvents);
  
  plotEntryType* plots_Embedded_reweighted2 = new plotEntryType("Embedded_reweighted2", true);
  TH1* lut_reweight2 = compRatioHistogram("lut_reweight2", plots_Ztautau->histogramGenDiTauPt_, plots_Embedded_reweighted1->histogramGenDiTauPt_);
  std::vector<weightEntryType*> weightEntries_Embedded_reweighted2;
  weightEntries_Embedded_reweighted2.push_back(new weightEntryType("Embedded_reweighted2", lut_reweight2, kGenDiTauPt));
  plots_Embedded_reweighted2->fillHistograms(tree_Embedded, weightEntries_Embedded_reweighted2, maxEvents);
  
  plotEntryType* plots_Embedded_reweighted3 = new plotEntryType("Embedded_reweighted3", true);
  TH2* lut_reweight3 = dynamic_cast<TH2*>(compRatioHistogram("lut_reweight3", plots_Ztautau->histogramGenTau2Pt_vs_GenTau1Pt_, plots_Embedded_reweighted2->histogramGenTau2Pt_vs_GenTau1Pt_));
  std::vector<weightEntryType*> weightEntries_Embedded_reweighted3;
  weightEntries_Embedded_reweighted3.push_back(new weightEntryType("Embedded_reweighted3", lut_reweight3, kGenTau1Pt, kGenTau2Pt));
  plots_Embedded_reweighted3->fillHistograms(tree_Embedded, weightEntries_Embedded_reweighted3, maxEvents);
  
  showDistribution(800, 900,
		   plots_Ztautau->histogramGenTau1Pt_, "gen. Z/#gamma^{*} #rightarrow #tau #tau",
		   plots_Embedded->histogramGenTau1Pt_, "rec. Embedding",
		   plots_Embedded_reweighted1->histogramGenTau1Pt_, "rec. Embedding, reweighted (1)",
		   plots_Embedded_reweighted2->histogramGenTau1Pt_, "rec. Embedding, reweighted (2)",
		   plots_Embedded_reweighted3->histogramGenTau1Pt_, "rec. Embedding, reweighted (3)",
		   0, "",
		   0., 250., 50, "P_{T}^{1} / GeV", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_genTau1Pt.pdf"));
  showDistribution(800, 900,
		   plots_Ztautau->histogramGenTau2Pt_, "gen. Z/#gamma^{*} #rightarrow #tau #tau",
		   plots_Embedded->histogramGenTau2Pt_, "rec. Embedding",
		   plots_Embedded_reweighted1->histogramGenTau2Pt_, "rec. Embedding, reweighted (1)",
		   plots_Embedded_reweighted2->histogramGenTau2Pt_, "rec. Embedding, reweighted (2)",
		   plots_Embedded_reweighted3->histogramGenTau2Pt_, "rec. Embedding, reweighted (3)",
		   0, "",
		   0., 250., 50, "P_{T}^{2} / GeV", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_genTau2Pt.pdf"));
  showDistribution(800, 900,
		   plots_Ztautau->histogramGenDiTauPt_, "gen. Z/#gamma^{*} #rightarrow #tau #tau",
		   plots_Embedded->histogramGenDiTauPt_, "rec. Embedding",
		   plots_Embedded_reweighted1->histogramGenDiTauPt_, "rec. Embedding, reweighted (1)",
		   plots_Embedded_reweighted2->histogramGenDiTauPt_, "rec. Embedding, reweighted (2)",
		   plots_Embedded_reweighted3->histogramGenDiTauPt_, "rec. Embedding, reweighted (3)",
		   0, "",
		   0., 250., 50, "M_{#tau#tau} / GeV", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_genDiTauPt.pdf"));
  showDistribution(800, 900,
		   plots_Ztautau->histogramGenDiTauMass_, "gen. Z/#gamma^{*} #rightarrow #tau #tau",
		   plots_Embedded->histogramGenDiTauMass_, "rec. Embedding",
		   plots_Embedded_reweighted1->histogramGenDiTauMass_, "rec. Embedding, reweighted (1)",
		   plots_Embedded_reweighted2->histogramGenDiTauMass_, "rec. Embedding, reweighted (2)",
		   plots_Embedded_reweighted3->histogramGenDiTauMass_, "rec. Embedding, reweighted (3)",
		   0, "",
		   0., 250., 50, "P_{T}^{#tau#tau} / GeV", 1.3,
		   true, 1.e-5, 1.e+1, "a.u", 1.3,
		   0.49, 0.71,
		   Form("plots/makeEmbeddingKineReweightLUTs2_genDiTauMass.pdf"));

  TString outputFileName = "makeEmbeddingKineReweightLUTs2.root";
  TFile* outputFile = TFile::Open(outputFileName.Data(), "RECREATE");
  plots_Ztautau->saveHistograms();
  plots_Embedded->saveHistograms();
  plots_Embedded_reweighted1->saveHistograms();
  plots_Embedded_reweighted2->saveHistograms();
  plots_Embedded_reweighted3->saveHistograms();
  lut_reweight1->Write();
  lut_reweight2->Write();
  lut_reweight3->Write();
  delete outputFile;

  delete inputFile_Ztautau;
  delete inputFile_Embedded;
}



