#include "CombinedCaloTowers.C"

void ProcessRelValRecHit(TFile &ref_file, TFile &val_file, ifstream &recstr, const int nHist1, const int nHist2, const int nProfInd, const int nHistTot, TString ref_vers, TString val_vers);

void RelValMacro(TString ref_vers="218", TString val_vers="218", TString rfname, TString vfname){

  ifstream RelValStream("InputRelVal.txt");
  
  TFile Ref_File(rfname); 
  TFile Val_File(vfname); 

  //Service variables
  //CaloTowers
  const int TTbar_CT_HB_nHist1 = 5;
  const int TTbar_CT_HE_nHist1 = 5;
  const int TTbar_CT_HF_nHist1 = 5;

  const int TTbar_CT_HB_nHistTot = 14;
  const int TTbar_CT_HE_nHistTot = 14;
  const int TTbar_CT_HF_nHistTot = 14;
  
  //RecHits
  const int TTbar_RH_nHistTot = 91; 
  const int TTbar_RH_nHist1   = 16;
  const int TTbar_RH_nHist2   = 4;
  const int TTbar_RH_nProfInd = 20;

  ProcessSubDetCT(Ref_File, Val_File, RelValStream, TTbar_CT_HB_nHist1, TTbar_CT_HB_nHistTot, ref_vers, val_vers);
  ProcessSubDetCT(Ref_File, Val_File, RelValStream, TTbar_CT_HE_nHist1, TTbar_CT_HE_nHistTot, ref_vers, val_vers);
  ProcessSubDetCT(Ref_File, Val_File, RelValStream, TTbar_CT_HF_nHist1, TTbar_CT_HF_nHistTot, ref_vers, val_vers);

  ProcessRelValRecHit(Ref_File, Val_File, RelValStream, TTbar_RH_nHist1, TTbar_RH_nHist2, TTbar_RH_nProfInd, TTbar_RH_nHistTot, ref_vers, val_vers);

  Ref_File.Close();
  Val_File.Close();

  return;
}

void ProcessRelValRecHit(TFile &ref_file, TFile &val_file, ifstream &recstr, const int nHist1, const int nHist2, const int nProfInd, const int nHistTot, TString ref_vers, TString val_vers){

  TCanvas* myc;

  TH1F*     ref_hist1[nHist1];
  TH2F*     ref_hist2[nHist2];
  TProfile* ref_prof[nProfInd];
  TH1D*     ref_fp[nProfInd];
 
  TH1F*     val_hist1[nHist1];
  TH2F*     val_hist2[nHist2];
  TProfile* val_prof[nProfInd];
  TH1D*     val_fp[nProfInd];

  //Workaround for ROOT bug: gPad must first be invoked outside
  //of "for" loop or one risks random failures
  gPad->SetLogy(0);
  
  int i;
  int DrawSwitch;
  TString StatSwitch, Chi2Switch, LogSwitch, DimSwitch;
  int RefCol, ValCol;
  TString HistName, HistName2;
  char xAxisTitle[200];
  float xAxisRange, yAxisRange, xMin;
  TString OutLabel;
  string xTitleCheck;

  int nh1 = 0;
  int nh2 = 0;
  int npr = 0;
  int npi = 0;
  
  for (i = 0; i < nHistTot; i++){
    //Read in 1/0 switch saying whether this histogram is used 
    //Skip it if not used, otherwise get output file label, histogram
    //axis ranges and title
    recstr>>HistName>>DrawSwitch;
    if (DrawSwitch == 0) continue;
    
    recstr>>OutLabel>>xAxisRange>>yAxisRange;
    recstr>>DimSwitch>>StatSwitch>>Chi2Switch>>LogSwitch;
    recstr>>RefCol>>ValCol;
    recstr.getline(xAxisTitle,200);
    
    //Format canvas
    if(DimSwitch == "PRwide") {
      gStyle->SetPadLeftMargin(0.06);
      gStyle->SetPadRightMargin(0.03);
      myc = new TCanvas("myc","",1200,600);
    }
    else myc = new TCanvas("myc","",800,600);
    myc->SetGrid();
    
    xTitleCheck = xAxisTitle;
    xTitleCheck = xTitleCheck.substr(1,7);

    //Format pad
    if (LogSwitch == "Log") gPad->SetLogy(1);
    else                    gPad->SetLogy(0);

    if (DimSwitch == "1D"){
      //Get histograms from files
      ref_file.cd("DQMData/HcalRecHitsV/HcalRecHitTask");   
      ref_hist1[nh1] = (TH1F*) gDirectory->Get(HistName);
      
      val_file.cd("DQMData/HcalRecHitsV/HcalRecHitTask");   
      val_hist1[nh1] = (TH1F*) gDirectory->Get(HistName);

      //Rebin histograms -- has to be done first
      if (Chi2Switch == "Chi2"){
	ref_hist1[nh1]->Rebin(10);
	val_hist1[nh1]->Rebin(10);
      }
      
      //Set the colors, styles, titles, stat boxes and format axes for the histograms 
      if (StatSwitch != "Stat" && StatSwitch != "Statrv") ref_hist1[nh1]->SetStats(kFALSE);   
      
      if (xAxisRange > 0){
	xMin = ref_hist1[nh1]->GetXaxis()->GetXmin();
	ref_hist1[nh1]->GetXaxis()->SetRangeUser(xMin,xAxisRange);
      }
      if (yAxisRange > 0) ref_hist1[nh1]->GetYaxis()->SetRangeUser(0.,yAxisRange);

      if (xTitleCheck != "NoTitle") ref_hist1[nh1]->GetXaxis()->SetTitle(xAxisTitle);

      //Different histo colors and styles
      ref_hist1[nh1]->SetTitle("");
      ref_hist1[nh1]->SetLineColor(RefCol);
      ref_hist1[nh1]->SetLineStyle(1); 
      ref_hist1[nh1]->SetMarkerSize(0.02);
      if (StatSwitch != "Stat" && StatSwitch != "Statrv") ref_hist1[nh1]->SetLineWidth(2); 
      
      val_hist1[nh1]->SetTitle("");
      val_hist1[nh1]->SetLineColor(ValCol);
      val_hist1[nh1]->SetLineStyle(2);  
      val_hist1[nh1]->SetMarkerSize(0.02);
      if (StatSwitch != "Stat" && StatSwitch != "Statrv") val_hist1[nh1]->SetLineWidth(2); 
   
      //Legend
      TLegend *leg = new TLegend(0.58, 0.91, 0.84, 0.99, "","brNDC");
      leg->SetBorderSize(2);
      leg->SetFillStyle(1001); 
      leg->AddEntry(ref_hist1[nh1],"CMSSW_"+ref_vers,"l");
      leg->AddEntry(val_hist1[nh1],"CMSSW_"+val_vers,"l");

      if (Chi2Switch == "Chi2"){
	//Draw and save histograms
	ref_hist1[nh1]->SetFillColor(48);
	ref_hist1[nh1]->Draw("hist");   
	val_hist1[nh1]->SetLineStyle(1);  
	if (StatSwitch == "Statrv") val_hist1[nh1]->Draw("hist sames e0");   
	else                        val_hist1[nh1]->Draw("hist same e0");   

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
	//Draw and save histograms
	ref_hist1[nh1]->Draw("hist");   
	if (StatSwitch == "Statrv") val_hist1[nh1]->Draw("hist sames");   
	else                        val_hist1[nh1]->Draw("hist same");   
      }

      //Stat Box where required
      if (StatSwitch == "Stat" || StatSwitch == "Statrv"){
	TPaveStats *ptstats = new TPaveStats(0.85,0.86,0.98,0.98,"brNDC");
	ptstats->SetTextColor(RefCol);
	ref_hist1[nh1]->GetListOfFunctions()->Add(ptstats);
	ptstats->SetParent(ref_hist1[nh1]->GetListOfFunctions());
	TPaveStats *ptstats = new TPaveStats(0.85,0.74,0.98,0.86,"brNDC");
	ptstats->SetTextColor(ValCol);
	val_hist1[nh1]->GetListOfFunctions()->Add(ptstats);
	ptstats->SetParent(val_hist1[nh1]->GetListOfFunctions());
	
	ptstats->Draw();
      }

      leg->Draw();   

      myc->SaveAs(OutLabel);
      nh1++;
    }     
    
    //Profiles not associated with histograms
    else if (DimSwitch == "PR" || DimSwitch == "PRwide"){
      //Get profiles from files
      ref_file.cd("DQMData/HcalRecHitsV/HcalRecHitTask");   
      ref_prof[npi] = (TProfile*) gDirectory->Get(HistName);
      
      val_file.cd("DQMData/HcalRecHitsV/HcalRecHitTask");   
      val_prof[npi] = (TProfile*) gDirectory->Get(HistName);
      
      //Legend
      TLegend *leg = new TLegend(0.58, 0.91, 0.84, 0.99, "","brNDC");
      leg->SetBorderSize(2);
      leg->SetFillStyle(1001); 

      //Ordinary profiles
      if (DimSwitch == "PR"){
	ref_prof[npi]->SetTitle("");
	ref_prof[npi]->SetErrorOption("");
	
	val_prof[npi]->SetTitle("");
	val_prof[npi]->SetErrorOption("");
	
	ref_prof[npi]->GetXaxis()->SetTitle(xAxisTitle);

	if (StatSwitch != "Stat" && StatSwitch != "Statrv"){
	  ref_prof[npi]->SetStats(kFALSE);   
	  val_prof[npi]->SetStats(kFALSE); 
	}

	ref_prof[npi]->SetLineColor(41);
	ref_prof[npi]->SetLineStyle(1);     
	ref_prof[npi]->SetLineWidth(1); 
	ref_prof[npi]->SetMarkerColor(41);
	ref_prof[npi]->SetMarkerStyle(21);
	ref_prof[npi]->SetMarkerSize(0.8);  
      
	val_prof[npi]->SetLineColor(43);
	val_prof[npi]->SetLineStyle(1);  
	val_prof[npi]->SetLineWidth(1); 
	val_prof[npi]->SetMarkerColor(43);
	val_prof[npi]->SetMarkerStyle(22);
	val_prof[npi]->SetMarkerSize(1.0);  

	ref_prof[npi]->Draw("hist pl");   
	val_prof[npi]->Draw("hist pl same");

	leg->AddEntry(ref_prof[npi],"CMSSW_"+ref_vers,"pl");
	leg->AddEntry(val_prof[npi],"CMSSW_"+val_vers,"pl");   	     
      }
      //Wide profiles
      else if (DimSwitch == "PRwide"){
	ref_fp[npi] = ref_prof[npi]->ProjectionX();    
	val_fp[npi] = val_prof[npi]->ProjectionX();
	
	ref_fp[npi]->SetTitle("");
	val_fp[npi]->SetTitle("");

	ref_fp[npi]->GetXaxis()->SetTitle(xAxisTitle);

	if (StatSwitch != "Stat" && StatSwitch != "Statrv"){
	  ref_fp[npi]->SetStats(kFALSE);   
	  val_fp[npi]->SetStats(kFALSE); 
	}   
	
	int nbins = ref_fp[npi]->GetNbinsX();
	for (int j = 1; j < nbins; j++) {
	  ref_fp[npi]->SetBinError(j, 0.);
	  val_fp[npi]->SetBinError(j, 0.);
	}
	ref_fp[npi]->SetLineWidth(0); 
	ref_fp[npi]->SetLineColor(0); // 5 yellow
	ref_fp[npi]->SetLineStyle(1); 
	ref_fp[npi]->SetMarkerColor(2);
	ref_fp[npi]->SetMarkerStyle(20);
	ref_fp[npi]->SetMarkerSize(0.5);
	
	val_fp[npi]->SetLineWidth(0); 
	val_fp[npi]->SetLineColor(0); // 45 blue
	val_fp[npi]->SetLineStyle(2); 
	val_fp[npi]->SetMarkerColor(4); 
	val_fp[npi]->SetMarkerStyle(22);
	val_fp[npi]->SetMarkerSize(0.5);
	
	ref_fp[npi]->Draw("p9");   
	val_fp[npi]->Draw("p9same");
	
	leg->AddEntry(ref_fp[npi],"CMSSW_"+ref_vers,"lp");
	leg->AddEntry(val_fp[npi],"CMSSW_"+val_vers,"lp");
	
      }
      
      leg->Draw("");   
      
      myc->SaveAs(OutLabel);
      
      npi++;
    }

    //Timing Histograms (special: read two lines at once)
    else if (DimSwitch == "TM"){
      
      recstr>>HistName2;

      ref_file.cd("DQMData/HcalRecHitsV/HcalRecHitTask");   
      
      ref_hist2[nh2] = (TH2F*) gDirectory->Get(HistName);
      ref_prof[npi]  = (TProfile*) gDirectory->Get(HistName2);
      
      val_file.cd("DQMData/HcalRecHitsV/HcalRecHitTask");   
      
      val_hist2[nh2] = (TH2F*) gDirectory->Get(HistName);
      val_prof[npi]  = (TProfile*) gDirectory->Get(HistName2);

      //Legend
      TLegend *leg = new TLegend(0.48, 0.91, 0.74, 0.99, "","brNDC");
      leg->SetBorderSize(2);
      leg->SetFillStyle(1001); 
      
      ref_hist2[nh2]->GetXaxis()->SetTitle(xAxisTitle);
      ref_hist2[nh2]->SetStats(kFALSE);
      
      ref_hist2[nh2]->SetMarkerColor(RefCol); // rose
      ref_hist2[nh2]->Draw();
      ref_prof[npi]->SetLineColor(41); 
      ref_prof[npi]->Draw("same");
      
      val_hist2[nh2]->SetMarkerColor(ValCol); 
      val_hist2[nh2]->Draw("same");
      val_prof[npi]->SetLineColor(45); 
      val_prof[npi]->Draw("same");
      
      leg->AddEntry(ref_prof[npi],"CMSSW_"+ref_vers,"pl");
      leg->AddEntry(val_prof[npi],"CMSSW_"+val_vers,"pl");   	     
      
      leg->Draw("");
      
      myc->SaveAs(OutLabel);    

      npi++;
      nh2++;
      i++;
    }
  }
  return;    
}
