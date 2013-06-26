//Auxiliary function
void ProcessSubDetCT(TFile &ref_file, TFile &val_file, ifstream &ctstr, const int nHist1, const int nHist2, const int nProf, const int nHistTot, TString ref_vers, TString val_vers, int harvest=0);

//Macro takes 3 parameters as arguments: the version to be validated, the reference version and a 2-bit integer that determines whether or not the samples are harvested
void CombinedCaloTowers(TString ref_vers="210", TString val_vers="210pre6", int harvest=0){
  
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
  const int HB_nHist1 = 7;
  const int HE_nHist1 = 7;
  const int HF_nHist1 = 7;

  const int HB_nHist2 = 2;
  const int HE_nHist2 = 2;
  const int HF_nHist2 = 2;

  const int HB_nProf = 2;
  const int HE_nProf = 2;
  const int HF_nProf = 2;

  const int HB_nHistTot = 20;
  const int HE_nHistTot = 20;
  const int HF_nHistTot = 20;

  //Order matters! InputCaloTowers.txt has histograms in the order HB-HE-HF
  ProcessSubDetCT(HB_ref_file, HB_val_file, CalTowStream, HB_nHist1, HB_nHist2, HB_nProf, HB_nHistTot, ref_vers, val_vers, harvest);
  ProcessSubDetCT(HE_ref_file, HE_val_file, CalTowStream, HE_nHist1, HE_nHist2, HE_nProf, HE_nHistTot, ref_vers, val_vers, harvest);
  ProcessSubDetCT(HF_ref_file, HF_val_file, CalTowStream, HF_nHist1, HF_nHist2, HE_nProf, HF_nHistTot, ref_vers, val_vers, harvest);

  //Close ROOT files
  HB_ref_file.Close();
  HE_ref_file.Close();
  HF_ref_file.Close();

  HB_val_file.Close();
  HE_val_file.Close();
  HF_val_file.Close();
  
  return;  
}

void ProcessSubDetCT(TFile &ref_file, TFile &val_file, ifstream &ctstr, const int nHist1, const int nHist2, const int nProf, const int nHistTot, TString ref_vers, TString val_vers, int harvest){
  
  TString RefHistDir, ValHistDir;

  if      (harvest == 11){
    RefHistDir = "DQMData/Run 1/CaloTowersV/Run summary/CaloTowersTask";
    ValHistDir = "DQMData/Run 1/CaloTowersV/Run summary/CaloTowersTask";
  }
  else if (harvest == 10){
    RefHistDir = "DQMData/CaloTowersV/CaloTowersTask";
    ValHistDir = "DQMData/Run 1/CaloTowersV/Run summary/CaloTowersTask";
  }
  else if (harvest == 1){
    RefHistDir = "DQMData/Run 1/CaloTowersV/Run summary/CaloTowersTask";
    ValHistDir = "DQMData/CaloTowersV/CaloTowersTask";
  }
  else{
    RefHistDir = "DQMData/CaloTowersV/CaloTowersTask";
    ValHistDir = "DQMData/CaloTowersV/CaloTowersTask";
  }
  
  TCanvas *myc = new TCanvas("myc","",800,600);
  
  TH1F* ref_hist1[nHist1];
  TH1F* val_hist1[nHist1];

  TH2F* ref_hist2[nHist2];
  TH2F* val_hist2[nHist2];
  
  TProfile* ref_prof[nProf];
  TProfile* val_prof[nProf];

  int i;
  int DrawSwitch;
  TString StatSwitch, Chi2Switch, LogSwitch, DimSwitch;
  int RefCol, ValCol;
  TString HistName, HistName2;
  char xAxisTitle[200];
  int nRebin;
  float xAxisMin, xAxisMax, yAxisMin, yAxisMax;
  TString OutLabel;
  
  int nh1 = 0;
  int nh2 = 0;
  int npi = 0;

  for (i = 0; i < nHistTot; i++){
    TLegend* leg = 0;
    TPaveText* ptchi2 = 0;
    TPaveStats *ptstats_r = 0;
    TPaveStats *ptstats_v = 0;

    //Read in 1/0 switch saying whether this histogram is used 
    //Skip it if not used, otherwise get output file label, histogram
    //axis ranges and title
    //Altered: Reads in all inputs and then uses 1/0 switch to skip
    ctstr>>HistName>>DrawSwitch;
//    if (DrawSwitch == 0) continue;
    
    ctstr>>OutLabel>>nRebin;
    ctstr>>xAxisMin>>xAxisMax>>yAxisMin>>yAxisMax;
    ctstr>>DimSwitch>>StatSwitch>>Chi2Switch>>LogSwitch;
    ctstr>>RefCol>>ValCol;
    ctstr.getline(xAxisTitle,200);
    if (DrawSwitch == 0) continue;
    
    //Format pad
    if (LogSwitch == "Log") myc->SetLogy();
    else                    myc->SetLogy(0);
    
    if (StatSwitch == "Stat") myc->SetGrid(0,0);
    else                      myc->SetGrid();
    
    if (DimSwitch == "1D"){
      //Get histograms from files
      ref_file.cd(RefHistDir);   
      ref_hist1[nh1] = (TH1F*) gDirectory->Get(HistName);

      ref_hist1[nh1]->SetMarkerStyle(21);       
      ref_hist1[nh1]->SetMarkerSize(0.02);

      val_file.cd(ValHistDir);   
      val_hist1[nh1] = (TH1F*) gDirectory->Get(HistName);

      val_hist1[nh1]->SetMarkerStyle(21);       
      val_hist1[nh1]->SetMarkerSize(0.02);


      // HACK to change what is embedded in DQM histos
      ref_hist1[nh1]->GetXaxis()->SetLabelSize(0.04); 
      val_hist1[nh1]->GetXaxis()->SetLabelSize(0.04); 
      ref_hist1[nh1]->GetYaxis()->SetLabelSize(0.04); 
      val_hist1[nh1]->GetYaxis()->SetLabelSize(0.04); 
      ref_hist1[nh1]->GetXaxis()->SetTitleSize(0.045); 
      val_hist1[nh1]->GetXaxis()->SetTitleSize(0.045); 

      ref_hist1[nh1]->GetXaxis()->SetTickLength(-0.015);
      val_hist1[nh1]->GetXaxis()->SetTickLength(-0.015);      
      ref_hist1[nh1]->GetYaxis()->SetTickLength(-0.015);
      val_hist1[nh1]->GetYaxis()->SetTickLength(-0.015);
      
      ref_hist1[nh1]->GetXaxis()->SetLabelOffset(0.02);
      val_hist1[nh1]->GetXaxis()->SetLabelOffset(0.02);
      ref_hist1[nh1]->GetYaxis()->SetLabelOffset(0.02);
      val_hist1[nh1]->GetYaxis()->SetLabelOffset(0.02);

      ref_hist1[nh1]->GetXaxis()->SetTitleOffset(1.0); 
      val_hist1[nh1]->GetXaxis()->SetTitleOffset(1.0); 

      
      //Rebin histograms -- has to be done first
      if (nRebin != 1){
	ref_hist1[nh1]->Rebin(nRebin);
	val_hist1[nh1]->Rebin(nRebin);
      }
      
      //Set the colors, styles, titles, stat boxes and format x-axis for the histograms 
      if (StatSwitch != "Stat"){
	ref_hist1[nh1]->SetStats(kFALSE);
	val_hist1[nh1]->SetStats(kFALSE);
      }
      else {
	ref_hist1[nh1]->SetStats(kTRUE);
	val_hist1[nh1]->SetStats(kTRUE);
      }

      //Min/Max Convetion: Default AxisMin = 0. Default AxisMax = -1.
      //xAxis
      if (xAxisMin == 0) xAxisMin = ref_hist1[nh1]->GetXaxis()->GetXmin();
      if (xAxisMax <  0) xAxisMax = ref_hist1[nh1]->GetXaxis()->GetXmax();
      
      if (xAxisMax > 0 || xAxisMin != 0){
	ref_hist1[nh1]->GetXaxis()->SetRangeUser(xAxisMin,xAxisMax);
	val_hist1[nh1]->GetXaxis()->SetRangeUser(xAxisMin,xAxisMax);
      }      
      //yAxis
      if (yAxisMin != 0) ref_hist1[nh1]->SetMinimum(yAxisMin);   
      if (yAxisMax  > 0) ref_hist1[nh1]->SetMaximum(yAxisMax);
      else if (ref_hist1[nh1]->GetMaximum() < val_hist1[nh1]->GetMaximum() &&
	       val_hist1[nh1]->GetMaximum() > 0){
	if (LogSwitch == "Log") ref_hist1[nh1]->SetMaximum(   2 * val_hist1[nh1]->GetMaximum());
	else                    ref_hist1[nh1]->SetMaximum(1.05 * val_hist1[nh1]->GetMaximum());
      }
      
      //Title
      ref_hist1[nh1]->GetXaxis()->SetTitle(xAxisTitle);
      
      //Different histo colors and styles
      ref_hist1[nh1]->SetTitle("");
      ref_hist1[nh1]->SetLineWidth(1); 
      ref_hist1[nh1]->SetLineColor(RefCol);
      ref_hist1[nh1]->SetLineStyle(1); 
      if (StatSwitch != "Stat") ref_hist1[nh1]->SetLineWidth(2); 
      
      val_hist1[nh1]->SetTitle("");
      val_hist1[nh1]->SetLineWidth(1); 
      val_hist1[nh1]->SetLineColor(ValCol);
      val_hist1[nh1]->SetLineStyle(2);
      if (StatSwitch != "Stat") val_hist1[nh1]->SetLineWidth(2); 
      
      //Chi2
      if (Chi2Switch == "Chi2"){
	//Draw histograms
	ref_hist1[nh1]->SetFillColor(40);//Originally 42. 
	ref_hist1[nh1]->Draw("hist"); // "stat"

	val_hist1[nh1]->SetLineStyle(1);
	val_hist1[nh1]->Draw("sames e0");
	//Get p-value from chi2 test
	const float NCHI2MIN = 0.01;
	
	float pval;
	stringstream mystream;
	char tempbuff[30];
      
	pval = ref_hist1[nh1]->Chi2Test(val_hist1[nh1]);
	
	sprintf(tempbuff,"Chi2 p-value: %6.3E%c",pval,'\0');
	mystream<<tempbuff;
	
	ptchi2 = new TPaveText(0.05,0.92,0.35,0.99, "NDC");
	
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
	ptstats_r = new TPaveStats(0.85,0.86,0.98,0.98,"brNDC");
	ptstats_r->SetTextColor(RefCol);
	ref_hist1[nh1]->GetListOfFunctions()->Add(ptstats_r);
	ptstats_r->SetParent(ref_hist1[nh1]->GetListOfFunctions());
	
	ptstats_v = new TPaveStats(0.85,0.74,0.98,0.86,"brNDC");
	ptstats_v->SetTextColor(ValCol);
	val_hist1[nh1]->GetListOfFunctions()->Add(ptstats_v);
	ptstats_v->SetParent(val_hist1[nh1]->GetListOfFunctions());
      }
      
      //Create legend
      leg = new TLegend(0.50, 0.91, 0.84, 0.99, "","brNDC");
      leg->SetBorderSize(2);
      leg->SetFillStyle(1001); //
      leg->AddEntry(ref_hist1[nh1],"CMSSW_"+ref_vers,"l");
      leg->AddEntry(val_hist1[nh1],"CMSSW_"+val_vers,"l");
      
      leg->Draw();   
      
      myc->SaveAs(OutLabel);
      
      nh1++;
    }

    else if (DimSwitch == "2D"){
      //Get histograms from files
      ref_file.cd(RefHistDir);   
      ref_hist2[nh2] = (TH2F*) gDirectory->Get(HistName);

      ref_hist2[nh2]->SetMarkerStyle(21);     
      ref_hist2[nh2]->SetMarkerSize(0.02);     

      val_file.cd(ValHistDir);   
      val_hist2[nh2] = (TH2F*) gDirectory->Get(HistName);

      val_hist2[nh2]->SetMarkerStyle(21);     
      val_hist2[nh2]->SetMarkerSize(0.02);     



      // HACK to change what is embedded in DQM histos
      ref_hist2[nh2]->GetXaxis()->SetLabelSize(0.04); 
      val_hist2[nh2]->GetXaxis()->SetLabelSize(0.04); 
      ref_hist2[nh2]->GetYaxis()->SetLabelSize(0.04); 
      val_hist2[nh2]->GetYaxis()->SetLabelSize(0.04); 
      ref_hist2[nh2]->GetXaxis()->SetTitleSize(0.045); 
      val_hist2[nh2]->GetXaxis()->SetTitleSize(0.045); 

      ref_hist2[nh2]->GetXaxis()->SetTickLength(-0.015);
      val_hist2[nh2]->GetXaxis()->SetTickLength(-0.015);      
      ref_hist2[nh2]->GetYaxis()->SetTickLength(-0.015);
      val_hist2[nh2]->GetYaxis()->SetTickLength(-0.015);
      
      ref_hist2[nh2]->GetXaxis()->SetLabelOffset(0.02);
      val_hist2[nh2]->GetXaxis()->SetLabelOffset(0.02);
      ref_hist2[nh2]->GetYaxis()->SetLabelOffset(0.02);
      val_hist2[nh2]->GetYaxis()->SetLabelOffset(0.02);

      ref_hist2[nh2]->GetXaxis()->SetTitleOffset(1.0); 
      val_hist2[nh2]->GetXaxis()->SetTitleOffset(1.0); 



      //Set the colors, styles, titles, stat boxes and format x-axis for the histograms 
      if (StatSwitch == "Stat") ref_hist2[nh2]->SetStats(kTRUE);
      
      //Min/Max Convetion: Default AxisMin = 0. Default AxisMax = -1.
      //xAxis
      if (xAxisMin == 0) xAxisMin = ref_hist2[nh2]->GetXaxis()->GetXmin();
      if (xAxisMax <  0) xAxisMax = ref_hist2[nh2]->GetXaxis()->GetXmax();
      
      if (xAxisMax > 0 || xAxisMin != 0){
	ref_hist2[nh2]->GetXaxis()->SetRangeUser(xAxisMin,xAxisMax);
	val_hist2[nh2]->GetXaxis()->SetRangeUser(xAxisMin,xAxisMax);
      }
      //yAxis
      if (yAxisMin != 0) ref_hist2[nh2]->SetMinimum(yAxisMin);   
      if (yAxisMax  > 0) ref_hist2[nh2]->SetMaximum(yAxisMax);
      else if (ref_hist2[nh2]->GetMaximum() < val_hist2[nh2]->GetMaximum() &&
	       val_hist2[nh2]->GetMaximum() > 0){
	if (LogSwitch == "Log") ref_hist2[nh2]->SetMaximum(   2 * val_hist2[nh2]->GetMaximum());
	else                    ref_hist2[nh2]->SetMaximum(1.05 * val_hist2[nh2]->GetMaximum());
      }
      
      //Legend
      leg = new TLegend(0.50, 0.91, 0.84, 0.99, "","brNDC");
      leg->SetBorderSize(2);
      leg->SetFillStyle(1001); 

      //Title
      ref_hist2[nh2]->GetXaxis()->SetTitle(xAxisTitle);
      ref_hist2[nh2]->SetStats(kFALSE);
      
      ref_hist2[nh2]->SetMarkerColor(RefCol); // rose
      ref_hist2[nh2]->Draw();
      
      val_hist2[nh2]->SetMarkerColor(ValCol); 
      val_hist2[nh2]->Draw("same");
      
      leg->AddEntry(ref_hist2[nh2],"CMSSW_"+ref_vers,"pl");
      leg->AddEntry(val_hist2[nh2],"CMSSW_"+val_vers,"pl");   	     
      
      leg->Draw("");
      
      myc->SaveAs(OutLabel);
      
      nh2++;
    }
    //Timing Histograms (special: read two lines at once)
    else if (DimSwitch == "TM"){
      
      ctstr>>HistName2;

      ref_file.cd(RefHistDir);   
      
      ref_hist2[nh2] = (TH2F*) gDirectory->Get(HistName);
      ref_prof[npi]  = (TProfile*) gDirectory->Get(HistName2);

      ref_hist2[nh2]->SetMarkerStyle(21);     
      ref_prof[npi] ->SetMarkerStyle(21);
      ref_hist2[nh2]->SetMarkerSize(0.02);     
      ref_prof[npi] ->SetMarkerSize(0.02);

      
      val_file.cd(ValHistDir);   
      
      val_hist2[nh2] = (TH2F*) gDirectory->Get(HistName);
      val_prof[npi]  = (TProfile*) gDirectory->Get(HistName2);

      val_hist2[nh2]->SetMarkerStyle(21);     
      val_prof[npi] ->SetMarkerStyle(21);
      val_hist2[nh2]->SetMarkerSize(0.02);     
      val_prof[npi] ->SetMarkerSize(0.02);


      // HACK to change what is embedded in DQM histos
      ref_hist2[nh2]->GetXaxis()->SetLabelSize(0.04); 
      val_hist2[nh2]->GetXaxis()->SetLabelSize(0.04); 
      ref_hist2[nh2]->GetYaxis()->SetLabelSize(0.04); 
      val_hist2[nh2]->GetYaxis()->SetLabelSize(0.04); 
      ref_hist2[nh2]->GetXaxis()->SetTitleSize(0.045); 
      val_hist2[nh2]->GetXaxis()->SetTitleSize(0.045); 

      ref_hist2[nh2]->GetXaxis()->SetTickLength(-0.015);
      val_hist2[nh2]->GetXaxis()->SetTickLength(-0.015);      
      ref_hist2[nh2]->GetYaxis()->SetTickLength(-0.015);
      val_hist2[nh2]->GetYaxis()->SetTickLength(-0.015);
      
      ref_hist2[nh2]->GetXaxis()->SetLabelOffset(0.02);
      val_hist2[nh2]->GetXaxis()->SetLabelOffset(0.02);
      ref_hist2[nh2]->GetYaxis()->SetLabelOffset(0.02);
      val_hist2[nh2]->GetYaxis()->SetLabelOffset(0.02);

      ref_hist2[nh2]->GetXaxis()->SetTitleOffset(1.0); 
      val_hist2[nh2]->GetXaxis()->SetTitleOffset(1.0); 


      //Min/Max Convetion: Default AxisMin = 0. Default AxisMax = -1.
      //xAxis
      if (xAxisMin == 0) xAxisMin = ref_hist2[nh2]->GetXaxis()->GetXmin();
      if (xAxisMax <  0) xAxisMax = ref_hist2[nh2]->GetXaxis()->GetXmax();

      if (xAxisMax > 0 || xAxisMin != 0){
	ref_hist2[nh2]->GetXaxis()->SetRangeUser(xAxisMin,xAxisMax);
	val_hist2[nh2]->GetXaxis()->SetRangeUser(xAxisMin,xAxisMax);
      }
      //yAxis
      if (yAxisMin != 0) ref_hist2[nh2]->SetMinimum(yAxisMin);   
      if (yAxisMax  > 0) ref_hist2[nh2]->SetMaximum(yAxisMax);  
      else if (ref_hist2[nh2]->GetMaximum() < val_hist2[nh2]->GetMaximum() &&
	       val_hist2[nh2]->GetMaximum() > 0){
	if (LogSwitch == "Log") ref_hist2[nh2]->SetMaximum(   2 * val_hist2[nh2]->GetMaximum());
	else                    ref_hist2[nh2]->SetMaximum(1.05 * val_hist2[nh2]->GetMaximum());
      }

      //Legend
      leg = new TLegend(0.50, 0.91, 0.84, 0.99, "","brNDC");
      leg->SetBorderSize(2);
      leg->SetFillStyle(1001); 
      
      ref_hist2[nh2]->GetXaxis()->SetTitle(xAxisTitle);
      ref_hist2[nh2]->SetStats(kFALSE);
     
      ref_hist2[nh2]->SetTitle("");
      val_prof[npi]->SetTitle("");

      ref_hist2[nh2]->SetMarkerColor(RefCol); // rose
      ref_hist2[nh2]->Draw();
      ref_prof[npi]->SetLineColor(RefCol); 
      ref_prof[npi]->Draw("same");
      
      val_hist2[nh2]->SetMarkerColor(ValCol); 
      val_hist2[nh2]->Draw("same");
      val_prof[npi]->SetLineColor(867); 
      val_prof[npi]->Draw("same");
      
      leg->AddEntry(ref_prof[npi],"CMSSW_"+ref_vers,"pl");
      leg->AddEntry(val_prof[npi],"CMSSW_"+val_vers,"pl");   	     
      
      leg->Draw("");
      
      myc->SaveAs(OutLabel);    

      npi++;
      nh2++;
      i++;
    }


    if(leg) delete leg;
    if(ptchi2) delete ptchi2;
    if(ptstats_r) delete ptstats_r;
    if(ptstats_v) delete ptstats_v;
  }
  if(myc) delete myc;
  return;
}
