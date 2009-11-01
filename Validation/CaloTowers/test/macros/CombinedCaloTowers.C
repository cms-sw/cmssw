//Auxiliary function
void ProcessSubDetCT(TFile &ref_file, TFile &val_file, ifstream &ctstr, const int nHist1, const int nHistTot, TString ref_vers, TString val_vers);

//Macro takes 2 parameters as arguments: the version to be validated and the reference version
void CombinedCaloTowers(TString ref_vers="210",
			TString val_vers="210pre6"){
  
  //Information contained in stream (in order): 
  //Name of histograms in root file, 1/0 switch whether they should be processed. If yes, then:
  //title of file where they should be saved, range of x-axis, range of y-axis (if not default)
  //Dimension of histogram, StatBox switch, Chi2 switch, logarithmic y-axis switch
  //color of reference histogram, color of validation histogram, title of histogram x-axis
  ifstream CalTowStream("InputCaloTowers.txt");
  
  //Specify files
  TFile HB_ref_file("HcalRecHitValidationHB_"+ref_vers+".root"); 
  TFile HE_ref_file("HcalRecHitValidationHE_"+ref_vers+".root"); 
  TFile HF_ref_file("HcalRecHitValidationHF_"+ref_vers+".root"); 

  TFile HB_val_file("HcalRecHitValidationHB_"+val_vers+".root");
  TFile HE_val_file("HcalRecHitValidationHE_"+val_vers+".root"); 
  TFile HF_val_file("HcalRecHitValidationHF_"+val_vers+".root"); 
  
  //Service variables
  const int HB_nHist1 = 5;
  const int HE_nHist1 = 5;
  const int HF_nHist1 = 5;

  const int HB_nHistTot = 14;
  const int HE_nHistTot = 14;
  const int HF_nHistTot = 14;

  //Order matters! InputCaloTowers.txt has histograms in the order HB-HE-HF
  ProcessSubDetCT(HB_ref_file, HB_val_file, CalTowStream, HB_nHist1, HB_nHistTot, ref_vers, val_vers);
  ProcessSubDetCT(HE_ref_file, HE_val_file, CalTowStream, HE_nHist1, HE_nHistTot, ref_vers, val_vers);
  ProcessSubDetCT(HF_ref_file, HF_val_file, CalTowStream, HF_nHist1, HF_nHistTot, ref_vers, val_vers);

  //Close ROOT files
  HB_ref_file.Close();
  HE_ref_file.Close();
  HF_ref_file.Close();

  HB_val_file.Close();
  HE_val_file.Close();
  HF_val_file.Close();
  
  return;  
}

void ProcessSubDetCT(TFile &ref_file, TFile &val_file, ifstream &ctstr, const int nHist1, const int nHistTot, TString ref_vers, TString val_vers){

  TCanvas *myc = new TCanvas("myc","",800,600);
  
  TH1F* ref_hist1[nHist1];
  TH1F* val_hist1[nHist1];

  //Workaround for ROOT bug: gPad must first be invoked outside
  //of "for" loop or one risks random failures
  gPad->SetLogy(0);
  
  int i;
  int DrawSwitch;
  TString StatSwitch, Chi2Switch, LogSwitch, DimSwitch;
  int RefCol, ValCol;
  TString HistName;
  char xAxisTitle[200];
  float xAxisRange, yAxisRange, xMin;
  TString OutLabel;
  
  int nh1 = 0;
  for (i = 0; i < nHistTot; i++){
    //Read in 1/0 switch saying whether this histogram is used 
    //Skip it if not used, otherwise get output file label, histogram
    //axis ranges and title
    ctstr>>HistName>>DrawSwitch;
    if (DrawSwitch == 0) continue;
    
    ctstr>>OutLabel>>xAxisRange>>yAxisRange;
    ctstr>>DimSwitch>>StatSwitch>>Chi2Switch>>LogSwitch;
    ctstr>>RefCol>>ValCol;
    ctstr.getline(xAxisTitle,200);
    
    //Format pad
    if (LogSwitch == "Log") gPad->SetLogy();
    else                    gPad->SetLogy(0);
    
    //Get histograms from files
    ref_file.cd("DQMData/CaloTowersV/CaloTowersTask");   
    ref_hist1[nh1] = (TH1F*) gDirectory->Get(HistName);

    val_file.cd("DQMData/CaloTowersV/CaloTowersTask");   
    val_hist1[nh1] = (TH1F*) gDirectory->Get(HistName);

    //Rebin histograms -- has to be done first
    if (Chi2Switch == "Chi2" && LogSwitch == "Log"){
	ref_hist1[nh1]->Rebin(5);
	val_hist1[nh1]->Rebin(5);
    }

    //Set the colors, styles, titles, stat boxes and format x-axis for the histograms 
    if (StatSwitch == "Stat") ref_hist1[nh1]->SetStats(kTRUE);

    if (xAxisRange > 0){
      xMin = ref_hist1[nh1]->GetXaxis()->GetXmin();
      ref_hist1[nh1]->GetXaxis()->SetRangeUser(xMin,xAxisRange);
    }
    if (yAxisRange > 0) ref_hist1[nh1]->GetYaxis()->SetRangeUser(0.,yAxisRange);

    ref_hist1[nh1]->GetXaxis()->SetTitle(xAxisTitle);
    
    //Different histo colors and styles
    ref_hist1[nh1]->SetTitle("");
    ref_hist1[nh1]->SetLineWidth(1); 
    ref_hist1[nh1]->SetLineColor(RefCol);
    //    ref_hist1[nh1]->SetLineStyle(1); 

    val_hist1[nh1]->SetTitle("");
    val_hist1[nh1]->SetLineWidth(1); 
    val_hist1[nh1]->SetLineColor(ValCol);
    //    val_hist1[nh1]->SetLineStyle(2);
    
    //Chi2
    if (Chi2Switch == "Chi2"){
      //Draw histograms
      ref_hist1[nh1]->SetFillColor(48);
      ref_hist1[nh1]->Draw("hist"); // "stat"
      val_hist1[nh1]->Draw("sames e0");
      //Get p-value from chi2 test
      const float NCHI2MIN = 0.01;
      
      float pval;
      stringstream mystream;
      char tempbuff[30];
      
      pval = ref_hist1[nh1]->Chi2Test(val_hist1[nh1]);

      sprintf(tempbuff,"Chi2 p-value: %6.3E%c",pval,'\0');
      mystream<<tempbuff;
      
      TPaveText* ptchi2 = new TPaveText(0.225,0.92,0.475,1.0, "NDC");
      
      if (pval > NCHI2MIN) ptchi2->SetFillColor(kGreen);
      else                 ptchi2->SetFillColor(kRed);
      
      ptchi2->SetTextSize(0.03);
      ptchi2->AddText(mystream.str().c_str());
      ptchi2->Draw();
    }
    else {
      //Draw histograms
      ref_hist1[nh1]->Draw("hist"); // "stat"
      val_hist1[nh1]->Draw("hist sames");
    }

    //StatBox
    if (StatSwitch == "Stat"){
      TPaveStats *ptstats = new TPaveStats(0.85,0.86,0.98,0.98,"brNDC");
      ptstats->SetTextColor(RefCol);
      ref_hist1[nh1]->GetListOfFunctions()->Add(ptstats);
      ptstats->SetParent(ref_hist1[nh1]->GetListOfFunctions());
      
      TPaveStats *ptstats = new TPaveStats(0.85,0.74,0.98,0.86,"brNDC");
      ptstats->SetTextColor(ValCol);
      val_hist1[nh1]->GetListOfFunctions()->Add(ptstats);
      ptstats->SetParent(val_hist1[nh1]->GetListOfFunctions());
    }
    
    //Create legend
    TLegend *leg = new TLegend(0.58, 0.91, 0.84, 0.99, "","brNDC");
    leg->SetBorderSize(2);
    leg->SetFillStyle(1001); //
    leg->AddEntry(ref_hist1[nh1],"CMSSW_"+ref_vers,"l");
    leg->AddEntry(val_hist1[nh1],"CMSSW_"+val_vers,"l");

    leg->Draw();   

    myc->SaveAs(OutLabel);
    
    nh1++;
  }
  return;
}
