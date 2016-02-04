//Auxiliary function
void ProcessSubDetDigi(TFile &ref_file, TFile &val_file, ifstream &digstr, const int nHist1, const int nHist2, const int nHistTot, TString ref_vers, TString val_vers);

//Macro takes 2 parameters as arguments: the version to be validated and the reference version
void CombinedDigis(TString ref_vers="220pre1",
		   TString val_vers="220"){
  
  //Information contained in stream (in order): 
  //Name of histograms in root file, 1/0 switch whether they should be processed. If yes, then:
  //title of file where they should be saved, range of x-axis, range of y-axis (if not default)
  //Dimension of histogram, StatBox switch, Chi2 switch, logarithmic y-axis switch
  //color of reference histogram, color of validation histogram, title of histogram x-axis
  ifstream DigStream("InputDigis.txt");

  //Specify files
  TFile HB_ref_file("HcalRecHitValidationHB_"+ref_vers+".root"); 
  TFile HE_ref_file("HcalRecHitValidationHE_"+ref_vers+".root");
  TFile HF_ref_file("HcalRecHitValidationHF_"+ref_vers+".root");
  TFile HF_gamma_ref_file("HcalRecHitValidationHF_gamma_"+ref_vers+".root");
  TFile HO_ref_file("HcalRecHitValidationHO_"+ref_vers+".root"); 
  TFile Noise_ref_file("HcalRecHitValidation_noise_NZS_"+ref_vers+".root"); 

  TFile HB_val_file("HcalRecHitValidationHB_"+val_vers+".root"); 
  TFile HE_val_file("HcalRecHitValidationHE_"+val_vers+".root"); 
  TFile HF_val_file("HcalRecHitValidationHF_"+val_vers+".root"); 
  TFile HF_gamma_val_file("HcalRecHitValidationHF_gamma_"+val_vers+".root"); 
  TFile HO_val_file("HcalRecHitValidationHO_"+val_vers+".root"); 
  TFile Noise_val_file("HcalRecHitValidation_noise_NZS_"+val_vers+".root"); 

  //Service variables
  const int HB_nHist1   = 5;
  const int HB_nHist2   = 1;
  const int HB_nHistTot = 35;

  const int HE_nHist1   = 7;
  const int HE_nHist2   = 2;
  const int HE_nHistTot = 35;

  const int HF_nHist1   = 5;
  const int HF_nHist2   = 2;
  const int HF_nHistTot = 35;

  const int HF_gamma_nHist1   = 5;
  const int HF_gamma_nHist2   = 2;
  const int HF_gamma_nHistTot = 35;

  const int HO_nHist1   = 3;
  const int HO_nHist2   = 1;
  const int HO_nHistTot = 35;

  const int Noise_nHist1   = 72;
  const int Noise_nHist2   = 0;
  const int Noise_nHistTot = 160;
  
  ProcessSubDetDigi(HB_ref_file, HB_val_file, DigStream, HB_nHist1, HB_nHist2, HB_nHistTot, ref_vers, val_vers);
  ProcessSubDetDigi(HE_ref_file, HE_val_file, DigStream, HE_nHist1, HE_nHist2, HE_nHistTot, ref_vers, val_vers);
  ProcessSubDetDigi(HF_ref_file, HF_val_file, DigStream, HF_nHist1, HF_nHist2, HF_nHistTot, ref_vers, val_vers);
  ProcessSubDetDigi(HF_gamma_ref_file, HF_gamma_val_file, DigStream, HF_gamma_nHist1, HF_gamma_nHist2, HF_gamma_nHistTot, ref_vers, val_vers);
  ProcessSubDetDigi(HO_ref_file, HO_val_file, DigStream, HO_nHist1, HO_nHist2, HO_nHistTot, ref_vers, val_vers);
  ProcessSubDetDigi(Noise_ref_file, Noise_val_file, DigStream, Noise_nHist1, Noise_nHist2, Noise_nHistTot, ref_vers, val_vers);

  //Close ROOT files
  HB_ref_file.Close();
  HE_ref_file.Close();
  HF_ref_file.Close();
  HF_gamma_ref_file.Close();
  HO_ref_file.Close();
  Noise_ref_file.Close();

  HB_val_file.Close();
  HE_val_file.Close();
  HF_val_file.Close();
  HF_gamma_val_file.Close();
  HO_val_file.Close();
  Noise_val_file.Close();

  return;
}

void ProcessSubDetDigi(TFile &ref_file, TFile &val_file, ifstream &digstr, const int nHist1, const int nHist2, const int nHistTot, TString ref_vers, TString val_vers){

  TCanvas *myc = new TCanvas("myc","",800,600);
  TLegend* leg = 0;
  TPaveText* ptchi2 = 0;
  TPaveStats *ptstats_r = 0;
  TPaveStats *ptstats_v = 0;
  
  TH1F* ref_hist1[nHist1];
  TH1F* val_hist1[nHist1];
  
  TH2F* ref_hist2[nHist2];
  TH2F* val_hist2[nHist2];
  
  int i;
  int DrawSwitch;
  TString StatSwitch, Chi2Switch, LogSwitch, DimSwitch;
  int RefCol, ValCol;
  TString HistName;
  char xAxisTitle[200];
  float xAxisRange, yAxisRange;
  TString OutLabel;
  string xTitleCheck;

  int nh1 = 0;
  int nh2 = 0;

  for (i = 0; i < nHistTot; i++){
    //Read in 1/0 switch saying whether this histogram is used 
    //Skip it if not used, otherwise get output file label, histogram
    //axis ranges and title
    digstr>>HistName>>DrawSwitch;
    if (DrawSwitch == 0) continue;
    
    digstr>>OutLabel>>xAxisRange>>yAxisRange;
    digstr>>DimSwitch>>StatSwitch>>Chi2Switch>>LogSwitch;
    digstr>>RefCol>>ValCol;
    digstr.getline(xAxisTitle,200);
    
    xTitleCheck = xAxisTitle;
    xTitleCheck = xTitleCheck.substr(1,7);

    //Format pad
    if (LogSwitch == "Log") myc->SetLogy();
    else                    myc->SetLogy(0);
    
    //1D Histo
    if (DimSwitch == "1D"){
      //Get histograms from files
      ref_file.cd("DQMData/HcalDigisV/HcalDigiTask");   
      ref_hist1[nh1] = (TH1F*) gDirectory->Get(HistName);
      
      val_file.cd("DQMData/HcalDigisV/HcalDigiTask");   
      val_hist1[nh1] = (TH1F*) gDirectory->Get(HistName);
      
      //Set the colors, styles, titles, stat boxes and format axes for the histograms 
      if (StatSwitch != "Stat" && StatSwitch != "Statrv") ref_hist1[nh1]->SetStats(kFALSE);   

      if (xAxisRange > 0){
	ref_hist1[nh1]->GetXaxis()->SetRangeUser(0.,xAxisRange);
	val_hist1[nh1]->GetXaxis()->SetRangeUser(0.,xAxisRange);
      }
      if (yAxisRange > 0) ref_hist1[nh1]->GetYaxis()->SetRangeUser(0.,yAxisRange);

      if (xTitleCheck != "NoTitle") ref_hist1[nh1]->GetXaxis()->SetTitle(xAxisTitle);

      //Different histo colors and styles
      ref_hist1[nh1]->SetTitle("");
      ref_hist1[nh1]->SetLineWidth(2); 
      ref_hist1[nh1]->SetLineColor(RefCol);
      ref_hist1[nh1]->SetLineStyle(1); 
      
      val_hist1[nh1]->SetTitle("");
      val_hist1[nh1]->SetLineWidth(3); 
      val_hist1[nh1]->SetLineColor(ValCol);
      val_hist1[nh1]->SetLineStyle(2);  
      
      //Legend
      leg = new TLegend(0.58, 0.91, 0.84, 0.99, "","brNDC");
      leg->SetBorderSize(2);
      leg->SetFillStyle(1001); //
      leg->AddEntry(ref_hist1[nh1],"CMSSW_"+ref_vers,"l");
      leg->AddEntry(val_hist1[nh1],"CMSSW_"+val_vers,"l");

      //Draw and save histograms
      ref_hist1[nh1]->Draw("hist");   
      if (StatSwitch == "Statrv") val_hist1[nh1]->Draw("hist sames");   
      else                        val_hist1[nh1]->Draw("hist same");   
      
      //Chi2
      if (Chi2Switch == "Chi2"){
	//Get p-value from chi2 test
	const float NCHI2MIN = 0.01;
	
	float pval;
	stringstream mystream;
	char tempbuff[30];
	
	pval = ref_hist1[nh1]->Chi2Test(val_hist1[nh1]);
      
	sprintf(tempbuff,"Chi2 p-value: %6.3E%c",pval,'\0');
	mystream<<tempbuff;
	
	ptchi2 = new TPaveText(0.225,0.92,0.475,1.0, "NDC");
	
	if (pval > NCHI2MIN) ptchi2->SetFillColor(kGreen);
	else                 ptchi2->SetFillColor(kRed);
	
	ptchi2->SetTextSize(0.03);
	ptchi2->AddText(mystream.str().c_str());
	ptchi2->Draw();
      }

      //Stat Box where required
      if (StatSwitch == "Stat" | StatSwitch == "Statrv"){
        ptstats_r = new TPaveStats(0.85,0.86,0.98,0.98,"brNDC");
        ptstats_r->SetTextColor(RefCol);
        ref_hist1[nh1]->GetListOfFunctions()->Add(ptstats_r);
        ptstats_r->SetParent(ref_hist1[nh1]->GetListOfFunctions());
        ptstats_v = new TPaveStats(0.85,0.74,0.98,0.86,"brNDC");
        ptstats_v->SetTextColor(ValCol);
        val_hist1[nh1]->GetListOfFunctions()->Add(ptstats_v);
        ptstats_v->SetParent(val_hist1[nh1]->GetListOfFunctions());
        
        ptstats_r->Draw();
        ptstats_v->Draw();
      }

      leg->Draw();   

      myc->SaveAs(OutLabel);
      nh1++;
    }     

    //2D Histo
    else if (DimSwitch == "2D"){
      //Get histograms from files
      ref_file.cd("DQMData/HcalDigisV/HcalDigiTask");   
      ref_hist2[nh2] = (TH2F*) gDirectory->Get(HistName);
      
      val_file.cd("DQMData/HcalDigisV/HcalDigiTask");   
      val_hist2[nh2] = (TH2F*) gDirectory->Get(HistName);

      //Set the colors, styles, titles, stat boxes and format axes for the histograms       
      if (StatSwitch != "Stat")	ref_hist2[nh2]->SetStats(kFALSE); 

      if (xAxisRange > 0){
	ref_hist2[nh2]->GetXaxis()->SetRangeUser(0.,xAxisRange);
	val_hist2[nh2]->GetXaxis()->SetRangeUser(0.,xAxisRange);
      }
      if (yAxisRange > 0) ref_hist2[nh2]->GetYaxis()->SetRangeUser(0.,yAxisRange);

      ref_hist2[nh2]->GetXaxis()->SetTitle(xAxisTitle);

      ref_hist2[nh2]->SetTitle("");
      ref_hist2[nh2]->SetLineColor(2);
      ref_hist2[nh2]->SetLineWidth(2); 
      ref_hist2[nh2]->SetMarkerColor(RefCol);
      ref_hist2[nh2]->SetMarkerStyle(20);
      ref_hist2[nh2]->SetMarkerSize(0.5);  
      
      val_hist2[nh2]->SetTitle("");
      val_hist2[nh2]->SetLineColor(3);
      val_hist2[nh2]->SetLineWidth(3); 
      val_hist2[nh2]->SetMarkerColor(ValCol);
      val_hist2[nh2]->SetMarkerStyle(22);
      val_hist2[nh2]->SetMarkerSize(0.5);  
      
      ref_hist2[nh2]->Draw("P");   
      val_hist2[nh2]->Draw("PSAME");   
      
      //Legend
      leg = new TLegend(0.58, 0.91, 0.84, 0.99, "","brNDC");
      leg->SetBorderSize(2);
      leg->SetFillStyle(1001); 
      leg->AddEntry(ref_hist2[nh2],"CMSSW_"+ref_vers,"pl");
      leg->AddEntry(val_hist2[nh2],"CMSSW_"+val_vers,"pl");
      
      leg->Draw();

      //Chi2
      if (Chi2Switch == "Chi2"){
	//Get p-value from chi2 test
	const float NCHI2MIN = 0.01;
	
	float pval;
	stringstream mystream;
	char tempbuff[30];
	
	pval = ref_hist2[nh2]->Chi2Test(val_hist2[nh2]);
      
	sprintf(tempbuff,"Chi2 p-value: %6.3E%c",pval,'\0');
	mystream<<tempbuff;
	
	ptchi2 = new TPaveText(0.225,0.92,0.475,1.0, "NDC");
	
	if (pval > NCHI2MIN) ptchi2->SetFillColor(kGreen);
	else                 ptchi2->SetFillColor(kRed);
	
	ptchi2->SetTextSize(0.03);
	ptchi2->AddText(mystream.str().c_str());
	ptchi2->Draw();
      }
      
      myc->SaveAs(OutLabel);
      nh2++;
    }
    if(leg) delete leg;
    if(ptchi2) delete ptchi2;
    if(ptstats_r) delete ptstats_r;
    if(ptstats_v) delete ptstats_v;
  }
  if(myc) delete myc;
  return;
}
