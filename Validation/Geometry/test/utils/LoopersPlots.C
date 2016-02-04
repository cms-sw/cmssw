TH1F* hist_bx_mean;
TH1F* hist_bx_95;

const Int_t nBins = 4;
const Int_t nBins_2 = 3;

void LoopersPlots() {
  setStyle();
  
  hist_bx_mean = new TH1F("hist_bx_mean","Bunch Crossing vs Pseudorapidity;Pseudorapidity |#eta|;Bunch Crossings [25 ns]",
			  10,0.0,1.0);
  hist_bx_95 = new TH1F("hist_bx_95","Bunch Crossing vs Pseudorapidity;Pseudorapidity |#eta|;Bunch Crossings [25 ns]",
			10,0.0,1.0);
  
  // fill with Loopers analalysis numbers
  Double_t Pseudorapidity[nBins]     = { 0.0      , 0.3      , 0.6      , 1.5      };
  Double_t BunchCrossing_Mean[nBins] = { 1.3105   , 0.752343 , 0.56186  , 0.614987 };
  Double_t BunchCrossing_RMS[nBins]  = { 0.851869 , 0.455688 , 0.289731 , 0.318793 };
  Double_t BunchCrossing_95[nBins]   = { 2.85     , 1.45     , 1.05     , 1.25     };
  
  for(Int_t i=0; i<nBins; i++) {
    // bunch crossings
    hist_bx_mean->Fill( Pseudorapidity[i] , BunchCrossing_Mean[i] );
    hist_bx_mean->SetBinError( hist_bx_mean->FindBin(Pseudorapidity[i]) , BunchCrossing_RMS[i] );
    hist_bx_95->Fill(   Pseudorapidity[i] , BunchCrossing_95[i]   );
  }

  hist_bx_mean_pT_eta00 = new TH1F("hist_bx_mean_pT_eta00",
				   "Bunch Crossing vs Transverse Momentum |#eta|=0;Transverse Momentum p_{T} [GeV/c];Bunch Crossings [25 ns]",
				   100,0.0,1.0);
  hist_bx_mean_pT_eta00->GetYaxis()->SetRangeUser(0.0,3.2);
  
  hist_bx_95_pT_eta00 = new TH1F("hist_bx_95_pT_eta00",
				 "Bunch Crossing vs Transverse Momentum |#eta|=0;Transverse Momentum p_{T} [GeV/c];Bunch Crossings [25 ns]",
				 100,0.0,1.0);
  hist_bx_95_pT_eta00->GetYaxis()->SetRangeUser(0.0,3.2);
  
  hist_bx_mean_pT_eta03 = new TH1F("hist_bx_mean_pT_eta03",
				   "Bunch Crossing vs Transverse Momentum |#eta|=0.3;Transverse Momentum p_{T} [GeV/c];Bunch Crossings [25 ns]",
				   100,0.0,1.0);
  hist_bx_mean_pT_eta03->GetYaxis()->SetRangeUser(0.0,3.2);
  
  hist_bx_95_pT_eta03 = new TH1F("hist_bx_95_pT_eta03",
				 "Bunch Crossing vs Transverse Momentum |#eta|=0.3;Transverse Momentum p_{T} [GeV/c];Bunch Crossings [25 ns]",
				 100,0.0,1.0);
  hist_bx_95_pT_eta03->GetYaxis()->SetRangeUser(0.0,3.2);
  
  // fill with Loopers analalysis numbers
  Double_t TransverseMomentum[nBins_2]          = { 0.2      , 0.4      , 0.6      };
  Double_t BunchCrossing_Mean_pT_eta00[nBins_2] = { 0.302835 , 0.702969 , 1.3105   };
  Double_t BunchCrossing_RMS_pT_eta00[nBins_2]  = { 0.135268 , 0.479322 , 0.851869 };
  Double_t BunchCrossing_95_pT_eta00[nBins_2]   = { 0.45     , 1.55     , 2.85     };
  Double_t BunchCrossing_Mean_pT_eta03[nBins_2] = { 0.264885 , 0.555507 , 0.752343 };
  Double_t BunchCrossing_RMS_pT_eta03[nBins_2]  = { 0.131494 , 0.388753 , 0.455688 };
  Double_t BunchCrossing_95_pT_eta03[nBins_2]   = { 0.45     , 1.35     , 1.45     };
  
  for(Int_t i=0; i<nBins_2; i++) {
    // bunch crossings
    hist_bx_mean_pT_eta00->Fill( TransverseMomentum[i] , BunchCrossing_Mean_pT_eta00[i] );
    hist_bx_mean_pT_eta00->SetBinError( hist_bx_mean_pT_eta00->FindBin(TransverseMomentum[i]) , BunchCrossing_RMS_pT_eta00[i] );
    hist_bx_95_pT_eta00->Fill(   TransverseMomentum[i] , BunchCrossing_95_pT_eta00[i]   );
    hist_bx_mean_pT_eta03->Fill( TransverseMomentum[i] , BunchCrossing_Mean_pT_eta03[i] );
    hist_bx_mean_pT_eta03->SetBinError( hist_bx_mean_pT_eta03->FindBin(TransverseMomentum[i]) , BunchCrossing_RMS_pT_eta03[i] );
    hist_bx_95_pT_eta03->Fill(   TransverseMomentum[i] , BunchCrossing_95_pT_eta03[i]   );
  }
  
  // draw
  drawHistos( hist_bx_95          , hist_bx_mean          , "bx"       );
  drawHistos( hist_bx_95_pT_eta00 , hist_bx_mean_pT_eta00 , "bx_eta00" );
  drawHistos( hist_bx_95_pT_eta03 , hist_bx_mean_pT_eta03 , "bx_eta03" );
  //
}

void drawHistos(TH1F* myHisto, TH1F* myHisto_err, TString myName) {
  //
  setStyle();
  //
  myHisto->SetMarkerColor(kBlue);
  myHisto->SetMarkerSize(1.5);
  myHisto->SetMarkerStyle(20);
  //  
  myHisto_err->SetMarkerColor(kRed);
  myHisto_err->SetMarkerSize(1.5);
  myHisto_err->SetMarkerStyle(21);
  //  
  
  // canvas
  TCanvas can("can","can",1000,900);
  can.Range(0,0,25,25);
  can.Divide(1,2);
  can.SetFillColor(kWhite);
  can.cd(1);
  //
  myHisto->GetXaxis()->SetNoExponent(1);
  myHisto->GetXaxis()->SetNdivisions(myHisto->GetNbinsX());
  myHisto->GetYaxis()->SetNdivisions(504);
  //
  myHisto->Draw("P,E2");
  //
  can.cd(2);
  //
  myHisto_err->GetXaxis()->SetNoExponent(1);
  myHisto_err->GetYaxis()->SetNoExponent(1);
  myHisto_err->GetXaxis()->SetNdivisions(myHisto_err->GetNbinsX());
  myHisto_err->GetYaxis()->SetNdivisions(504);
  //
  myHisto_err->Draw("P");
  //
  //
  can.Update();
  can.SaveAs( Form( "LoopersPlots_%s.eps", myName.Data() ) );
  can.SaveAs( Form( "LoopersPlots_%s.gif", myName.Data() ) );
  can.SaveAs( Form( "LoopersPlots_%s.C"  , myName.Data() ) );
  //
}

void drawHistos(TGraphAsymmErrors* myHisto, TH1F* myHisto_low, TH1F* myHisto_high , TString myName) {
  //
  setStyle();
  //
  myHisto->SetMarkerColor(kBlue);
  myHisto->SetMarkerSize(1.5);
  myHisto->SetMarkerStyle(20);
  myHisto->SetLineWidth(2);
  //
  myHisto_low->SetFillColor(kWhite);
  myHisto_low->SetLineColor(kGreen);
  myHisto_low->SetLineWidth(4);
  //  
  myHisto_high->SetFillColor(kWhite);
  myHisto_high->SetLineColor(kRed);
  myHisto_high->SetLineWidth(4);
  //  
  // canvas
  TCanvas can("can","can",1000,800);
  can.Range(0,0,25,25);
  can.SetFillColor(kWhite);
  can.cd();
  //
  myHisto->GetXaxis()->SetNoExponent(1);
  myHisto->GetXaxis()->SetNdivisions(myHisto->GetN());
  myHisto->GetYaxis()->SetNdivisions(504);
  //
  myHisto->Draw("AP,E1");
  myHisto_low->Draw("L,SAME");
  myHisto_high->Draw("L,SAME");
  //
  // Legenda
  TLegend* theLegend = new TLegend(0.70, 0.70, 0.89, 0.89);
  theLegend->AddEntry( myHisto_low  , "95% CL lower limit" , "L" );
  theLegend->AddEntry( myHisto      , "EW Fit"             , "P" );
  theLegend->AddEntry( myHisto_high , "EW Fit upper limit" , "L" );
  theLegend->Draw();
  //
  can.Update();
  can.SaveAs( Form( "LoopersPlots_%s.eps", myName.Data() ) );
  can.SaveAs( Form( "LoopersPlots_%s.gif", myName.Data() ) );
  can.SaveAs( Form( "LoopersPlots_%s.C"  , myName.Data() ) );
  //
}

void drawHistos(TH1F* myHisto_1, TH1F* myHisto_2 , TString myName) {
  //
  setStyle();
  //
  myHisto_1->SetFillColor(kWhite);
  myHisto_1->SetLineColor(kBlack);
  myHisto_1->SetLineWidth(2);
  myHisto_1->SetMarkerStyle(20);
  myHisto_1->SetMarkerColor(kBlue);
  //  
  myHisto_2->SetFillColor(kWhite);
  myHisto_2->SetLineColor(kBlack);
  myHisto_2->SetLineWidth(2);
  myHisto_2->SetMarkerStyle(21);
  myHisto_2->SetMarkerColor(kRed);
  //  
  // canvas
  TCanvas can("can","can",1000,800);
  can.Range(0,0,25,25);
  can.SetFillColor(kWhite);
  can.cd();
  //
  //  myHisto_mean->GetXaxis()->SetNoExponent(1);
  //  myHisto_mean->GetXaxis()->SetNdivisions(myHisto->GetN());
  //  myHisto_mean->GetYaxis()->SetNdivisions(504);
  //
  myHisto_1->Draw("P");
  myHisto_2->Draw("PE1,SAME");
  //
  // Legenda
  TLegend* theLegend = new TLegend(0.70, 0.70, 0.89, 0.89);
  theLegend->AddEntry( myHisto_1 , "95%" , "P" );
  theLegend->AddEntry( myHisto_2 , "Mean"  , "P" );
  theLegend->Draw();
  //
  can.Update();
  can.SaveAs( Form( "LoopersPlots_%s.eps", myName.Data() ) );
  can.SaveAs( Form( "LoopersPlots_%s.gif", myName.Data() ) );
  can.SaveAs( Form( "LoopersPlots_%s.C"  , myName.Data() ) );
  //
}

void setStyle(){
  // Style
  gStyle->SetOptStat(0000);
  gStyle->SetOptFit(1111);
  gStyle->SetFuncColor(kRed);
  //
  gStyle->SetLabelSize(0.04,"x,y,z");
  gStyle->SetTitleSize(0.05,"x,y,z");
  gStyle->SetTitleOffset(0.8,"x,y,z");
  gStyle->SetTitleFontSize(0.06);
  //
  //  gStyle->SetPadLeftMargin(0.10);
  //  gStyle->SetPadBottomMargin(0.10);
  //
  gStyle->SetHistLineWidth(1);
  //
  gStyle->SetPaintTextFormat("g");
  //
  gStyle->SetTitleBorderSize(0);
  gStyle->SetTitleFillColor(0);
  gStyle->SetTitleFont(12,"pad");
  gStyle->SetTitleFontSize(0.04);
  gStyle->SetTitleX(0.075);
  gStyle->SetTitleY(0.950);
  //
}

