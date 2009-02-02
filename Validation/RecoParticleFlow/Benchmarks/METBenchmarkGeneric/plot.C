double fitf(double *x, double *par)
{
  double fitval = sqrt( par[0]*par[0] + 
			par[1]*par[1]*(x[0]-par[3]) + 
			par[2]*par[2]*(x[0]-par[3])*(x[0]-par[3]) );
  return fitval;
}

void FitSlicesInY(TH2F* h, TH1F* mean, TH1F* sigma, bool doGausFit, int type )
{
  TAxis *fXaxis = h->GetXaxis();
  TAxis *fYaxis = h->GetYaxis();
  Int_t nbins  = fXaxis->GetNbins();
  Int_t binmin = 1;
  Int_t binmax = nbins;
  TString option = "QNR";
  TString opt = option;
  opt.ToLower();
  Float_t ngroup = 1;
  ngroup = 1;

  //default is to fit with a gaussian
  TF1 *f1 = 0;
  if (f1 == 0) 
    {
      //f1 = (TF1*)gROOT->GetFunction("gaus");
      if (f1 == 0) f1 = new TF1("gaus","gaus", fYaxis->GetXmin(), fYaxis->GetXmax());
      else         f1->SetRange( fYaxis->GetXmin(), fYaxis->GetXmax());
    }
  Int_t npar = f1->GetNpar();
  if (npar <= 0) return;
  Double_t *parsave = new Double_t[npar];
  f1->GetParameters(parsave);

  //Create one histogram for each function parameter
  Int_t ipar;
  TH1F **hlist = new TH1F*[npar];
  char *name   = new char[2000];
  char *title  = new char[2000];
  const TArrayD *bins = fXaxis->GetXbins();
  for( ipar=0; ipar < npar; ipar++ ) 
    {
      if( ipar == 1 ) 
	if( type == 1 )   sprintf(name,"meanPF");
	else              sprintf(name,"meanCalo");
      else
	if( doGausFit ) 
	  if( type == 1 ) sprintf(name,"sigmaPF");
	  else            sprintf(name,"sigmaCalo");
	else 
	  if( type == 1 ) sprintf(name,"rmsPF");
	  else            sprintf(name,"rmsCalo");
      if( type == 1 )     sprintf(title,"Particle Flow");
      else                sprintf(title,"Calorimeter");
      delete gDirectory->FindObject(name);
      if (bins->fN == 0) 
	hlist[ipar] = new TH1F(name,title, nbins, fXaxis->GetXmin(), fXaxis->GetXmax());
      else
	hlist[ipar] = new TH1F(name,title, nbins,bins->fArray);
      hlist[ipar]->GetXaxis()->SetTitle(fXaxis->GetTitle());
    }
  sprintf(name,"test_chi2");
  delete gDirectory->FindObject(name);
  TH1F *hchi2 = new TH1F(name,"chisquare", nbins, fXaxis->GetXmin(), fXaxis->GetXmax());
  hchi2->GetXaxis()->SetTitle(fXaxis->GetTitle());

  //Loop on all bins in X, generate a projection along Y
  Int_t bin;
  Int_t nentries;
  for( bin = (Int_t) binmin; bin <= (Int_t) binmax; bin += ngroup ) 
    {
      TH1F *hpy = (TH1F*) h->ProjectionY("_temp", (Int_t) bin, (Int_t) bin + ngroup - 1, "e");
      if(hpy == 0) continue;
      nentries = Int_t( hpy->GetEntries() );
      if(nentries == 0 ) {delete hpy; continue;}
      f1->SetParameters(parsave);
      hpy->Fit( f1, opt.Data() );
      Int_t npfits = f1->GetNumberFitPoints(); 
      //cout << "bin = " << bin << "; Npfits = " << npfits << "; npar = " << npar << endl;
      if( npfits > npar ) 
	{
	  Int_t biny = bin + ngroup/2;
	  for( ipar=0; ipar < npar; ipar++ ) 
	    {
	      if( doGausFit ) hlist[ipar]->Fill( fXaxis->GetBinCenter(biny), f1->GetParameter(ipar) );
	      else            hlist[ipar]->Fill( fXaxis->GetBinCenter(biny), hpy->GetRMS() );
	      //cout << "bin[" << bin << "]: RMS = " << hpy->GetRMS() << "; sigma = " << f1->GetParameter(ipar) << endl;
	      hlist[ipar]->SetBinError( biny, f1->GetParError(ipar) );
	    }
	  hchi2->Fill( fXaxis->GetBinCenter(biny), f1->GetChisquare()/(npfits-npar) );
	}
      delete hpy;
      ngroup += ngroup*0.2;//0.1  //used for non-uniform binning
    }
  *mean = *hlist[1];
  *sigma = *hlist[2];
  //cout << "Entries = " << hlist[0]->GetEntries() << endl;
}

void plot(void)
{
// here analyze your histograms, to compare PF and calo MET. 
// this is the example of ../JetBenchmarkGeneric

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

TFile f("metBenchmarkGeneric.root");


TCanvas c1;
FormatPad( &c1, false );

f.cd("DQMData/PFTask/Benchmarks/pfMet/Gen");

//TH2F* hpf = (TH2F*) gDirectory.Get("SETvsDeltaMET");
TH2F* hpf = (TH2F*) gROOT->FindObject("SETvsDeltaMET");
hpf->RebinX(16);
hpf->RebinY(2);
FormatHisto(hpf, s1);
hpf->Draw();

hpf->FitSlicesY();

TH1* hpf_1 = (TH1*) gROOT->FindObject("MEX");
FormatHisto(hpf_1, s2);
hpf_1->Draw("same");
//hpf_1->Draw();

gPad->SaveAs("test1.png");

TCanvas c2;
FormatPad( &c2, false );
TH1D* hpfpy = hpf->ProjectionY();
FormatHisto(hpfpy, s1);
hpfpy.Draw();

gPad->SaveAs("test2.png");

  //Define fit functions and histograms
  TF1* func1 = new TF1("fit1", fitf, 0, 40, 4);
  TF1* func2 = new TF1("fit2", fitf, 0, 40, 4);
  TF1* func3 = new TF1("fit3", fitf, 0, 40, 4);
  TF1* func4 = new TF1("fit4", fitf, 0, 40, 4);

  TH2F* hSETvsDeltaMET = (TH2F*) gROOT->FindObject("SETvsDeltaMET");
  //TH1F* hmeanPF = (TH1F*) gROOT->FindObject("meanPF");
  TH1F* hmeanPF  = new TH1F("meanPF",  "Mean PFMEX",    100, 0.0, 1600.0);
  TH1F* hrmsPF = (TH1F*) gROOT->FindObject("rmsPF");
  //TH1F* hsigmaPF = (TH1F*) gROOT->FindObject("sigmaPF");
  TH1F* hsigmaPF = new TH1F("sigmaPF", "#sigma(PFMEX)", 100, 0.0, 1600.0);
  
  //fit gaussian to Delta MET corresponding to different slices in MET, store fit values (mean,sigma) in histos
  FitSlicesInY(hSETvsDeltaMET, hmeanPF, hsigmaPF, false, 1); //set option flag for RMS or gaussian
  //FitSlicesInY(hSETvsDeltaMET, hmeanPF, hsigmaPF, true, 1); //set option flag for RMS or gaussian
//FitSlicesInY(hCaloSETvsDeltaCaloMET, hmeanCalo, hrmsCalo, false, 2); 
//FitSlicesInY(hCaloSETvsDeltaCaloMET, hmeanCalo, hsigmaCalo, true, 2); 

/*
  SETAXES(meanPF,    "SET", "Mean(MEX)");
  SETAXES(meanCalo,  "SET", "Mean(MEX)");
  SETAXES(sigmaPF,   "SET", "#sigma(MEX)");
  SETAXES(sigmaCalo, "SET", "#sigma(MEX)");
  SETAXES(rmsPF,     "SET", "RMS(MEX)");
  SETAXES(rmsCalo,   "SET", "RMS(MEX)");
*/
  // Make the MET resolution versus SET plot
  
  TCanvas* canvas_MetResVsRecoSet = new TCanvas("MetResVsRecoSet", "MET Sigma vs Reco SET", 500,500);
  hsigmaPF->SetStats(0); 
  func1->SetLineColor(1); 
  func1->SetParNames("Noise", "Stochastic", "Constant", "Offset");
  func1->SetParameters(10.0, 0.8, 0.1, 100.0);
  hsigmaPF->Fit("fit1", "", "", 100.0, 900.0);
  /*
  func2->SetLineColor(2); 
  func2->SetParNames("Noise", "Stochastic", "Constant", "Offset");
  func2->SetParameters(10.0, 0.8, 0.1, 100.0);
  hsigmaCalo->Fit("fit2", "", "", 100.0, 900.0);
  func3->SetLineColor(4); 
  func3->SetParNames("Noise", "Stochastic", "Constant", "Offset");
  func3->SetParameters(10.0, 0.8, 0.1, 100.0);
  hrmsPF->Fit("fit3", "", "", 100.0, 900.0);
  func4->SetLineColor(6); 
  func4->SetParNames("Noise", "Stochastic", "Constant", "Offset");
  func4->SetParameters(10.0, 0.8, 0.1, 100.0);
  hrmsCalo->Fit("fit4", "", "", 100.0, 900.0);
  */
  (hsigmaPF->GetYaxis())->SetRangeUser( 0.0, 50.0);
  hsigmaPF->SetLineWidth(2); 
  hsigmaPF->SetLineColor(1); 
  hsigmaPF->Draw();
  /*
  hsigmaCalo->SetLineWidth(2);
  hsigmaCalo->SetLineColor(2);
  hsigmaCalo->Draw("SAME");
  hrmsPF->SetLineWidth(2);
  hrmsPF->SetLineColor(4);
  hrmsPF->Draw("SAME");  
  hrmsCalo->SetLineWidth(2);
  hrmsCalo->SetLineColor(6);
  hrmsCalo->Draw("SAME");
  */

  // Make the SET response versus SET plot
  /*
  TCanvas* canvas_SetRespVsTrueSet = new TCanvas("SetRespVsTrueSet", "SET Response vs True SET", 500,500);
  profileSETvsSETresp->SetStats(0); 
  profileSETvsSETresp->SetStats(0); 
  (profileSETvsSETresp->GetYaxis())->SetRangeUser(-1.0, 1.0);
  profileSETvsSETresp->SetLineWidth(2); 
  profileSETvsSETresp->SetLineColor(4); 
  profileSETvsSETresp->Draw();
  profileCaloSETvsCaloSETresp->SetLineWidth(2); 
  profileCaloSETvsCaloSETresp->SetLineColor(2); 
  profileCaloSETvsCaloSETresp->Draw("SAME");
  */

  // Make the MET response versus MET plot
  /*
  TCanvas* canvas_MetRespVsTrueMet = new TCanvas("MetRespVsTrueMet", "MET Response vs True MET", 500,500);
  profileMETvsMETresp->SetStats(0); 
  profileMETvsMETresp->SetStats(0); 
  (profileMETvsMETresp->GetYaxis())->SetRangeUser(-1.0, 1.0);
  profileMETvsMETresp->SetLineWidth(2); 
  profileMETvsMETresp->SetLineColor(4); 
  profileMETvsMETresp->Draw();
  profileCaloMETvsCaloMETresp->SetLineWidth(2); 
  profileCaloMETvsCaloMETresp->SetLineColor(2); 
  profileCaloMETvsCaloMETresp->Draw("SAME");
  */

  //print the resulting plots to file
  /*
  canvas_MetResVsRecoSet->Print("MetResVsRecoSet.ps");
  canvas_SetRespVsTrueSet->Print("SetRespVsTrueSet.ps");
  canvas_MetRespVsTrueMet->Print("MetRespVsTrueMet.ps");  
  */

}

