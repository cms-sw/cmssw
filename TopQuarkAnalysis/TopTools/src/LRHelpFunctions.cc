#include "TopQuarkAnalysis/TopTools/interface/LRHelpFunctions.h"
#include "TopQuarkAnalysis/TopTools/test/tdrstyle.C"

// constructors
LRHelpFunctions::LRHelpFunctions() {
  constructPurity = false;
  setTDRStyle();
  gStyle->SetCanvasDefW(900);
}


LRHelpFunctions::LRHelpFunctions(const std::vector<int>& obsNr, int nrBins, const std::vector<double>& obsMin, const std::vector<double>& obsMax, const std::vector<const char*>& functions) { 
  obsNumbers = obsNr;
  constructPurity = false;
  setTDRStyle();
  gStyle->SetCanvasDefW(900);
  for(size_t o=0; o<obsNr.size(); o++){
    // create Signal, background and s/(s+b) histogram
    TString htS  = "Obs"; htS  += obsNr[o]; htS += "_S"; 
    TString htB  = "Obs"; htB  += obsNr[o]; htB += "_B"; 
    TString htSB = "Obs"; htSB += obsNr[o]; htSB += "_SoverSplusB"; 
    hObsS.push_back( new TH1F(htS,htS,nrBins,obsMin[o],obsMax[o]) );
    hObsB.push_back( new TH1F(htB,htB,nrBins,obsMin[o],obsMax[o]) );
    hObsSoverSplusB.push_back( new TH1F(htSB,htSB,nrBins,obsMin[o],obsMax[o]) );
    
    // create the correlation 2D plots among the observables (for signal)
    for (size_t o2 = o+1; o2 < obsNr.size(); o2++) {
      TString hcorr  = "Corr_Obs"; hcorr  += obsNr[o]; hcorr += "_Obs";  hcorr  += obsNr[o2]; 
      hObsCorr.push_back( new TH2F(hcorr,hcorr,nrBins,obsMin[o],obsMax[o],nrBins,obsMin[o2],obsMax[o2]) );
    }    
    
    // create fit functions
    TString ftSB = "F_Obs"; ftSB += obsNr[o]; ftSB += "_SoverSplusB"; 
    fObsSoverSplusB.push_back( new TF1(ftSB,functions[o],hObsS[o]->GetXaxis()->GetXmin(),hObsS[o]->GetXaxis()->GetXmax()) );
  }
}

void LRHelpFunctions::recreateFitFct(const std::vector<int>& obsNr, const std::vector<const char*>& functions) {
  if (!fObsSoverSplusB.empty()) fObsSoverSplusB.clear();
  for(size_t o=0; o<obsNr.size(); o++){
    // create fit functions
    TString ftSB = "F_Obs"; ftSB += obsNr[o]; ftSB += "_SoverSplusB";
    fObsSoverSplusB.push_back( new TF1(ftSB,functions[o],hObsS[o]->GetXaxis()->GetXmin(),hObsS[o]->GetXaxis()->GetXmax()) );
  }
}

LRHelpFunctions::LRHelpFunctions(int nrLRbins, double LRmin, double LRmax, const char* LRfunction) { 
  constructPurity = true;
  setTDRStyle();
  gStyle->SetCanvasDefW(900);

  // create LR histograms
  hLRtotS = new TH1F("hLRtotS","hLRtotS",nrLRbins,LRmin,LRmax);
  hLRtotS->GetXaxis()->SetTitle("Combined LR");
  hLRtotB = new TH1F("hLRtotB","hLRtotB",nrLRbins,LRmin,LRmax);
  hLRtotB->GetXaxis()->SetTitle("Combined LR");
  hLRtotSoverSplusB = new TH1F("hLRtotSoverSplusB","hLRtotSoverSplusB",nrLRbins,LRmin,LRmax);
    
  // create LR fit function
  fLRtotSoverSplusB = new TF1("fLRtotSoverSplusB",LRfunction,LRmin,LRmax);
}


// destructor
LRHelpFunctions::~LRHelpFunctions() {}


// member function to initialize the LR hists and fits
void LRHelpFunctions::initLRHistsAndFits(int nrLRbins, double LRmin, double LRmax, const char* LRfunction){ 
  constructPurity = true;
  // create LR histograms
  hLRtotS = new TH1F("hLRtotS","hLRtotS",nrLRbins,LRmin,LRmax);
  hLRtotB = new TH1F("hLRtotB","hLRtotB",nrLRbins,LRmin,LRmax);
  hLRtotSoverSplusB = new TH1F("hLRtotSoverSplusB","hLRtotSoverSplusB",nrLRbins,LRmin,LRmax);    
  // create LR fit function
  fLRtotSoverSplusB = new TF1("fLRtotSoverSplusB",LRfunction,LRmin,LRmax);
}



// member function to set initial values to the observable fit function
void LRHelpFunctions::setObsFitParameters(int obs,const std::vector<double>& fitPars){
  for(size_t fit=0; fit<fObsSoverSplusB.size(); fit++){
    TString fn = "_Obs"; fn += obs;
    if(((TString)fObsSoverSplusB[fit]->GetName()).Contains(fn)){
      for(size_t p=0; p<fitPars.size(); p++){
        fObsSoverSplusB[fit]->SetParameter(p,fitPars[p]);
      }
    }
  }
}





// member function to add observable values to the signal histograms
void LRHelpFunctions::fillToSignalHists(const std::vector<double>& obsVals, double weight){
  int hIndex = 0;
  for(size_t o=0; o<obsVals.size(); o++) {
    hObsS[o]->Fill(obsVals[o], weight);
    for(size_t o2=o+1; o2<obsVals.size(); o2++) {
      hObsCorr[hIndex] -> Fill(obsVals[o],obsVals[o2], weight);
      ++hIndex;
    }
  }
}

// member function to add observable values to the signal histograms
void LRHelpFunctions::fillToSignalHists(int obsNbr, double obsVal, double weight){
  TString obs = "Obs"; obs += obsNbr; obs += "_";
  for(size_t f = 0; f<hObsS.size(); f++){
    if(((TString)(hObsS[f]->GetName())).Contains(obs)) {
      hObsS[f]->Fill(obsVal, weight);
      return;
    }
  }
}

// member function to add observable values to the signal histograms
void LRHelpFunctions::fillToSignalCorrelation(int obsNbr1, double obsVal1, int obsNbr2,
	double obsVal2, double weight){
  TString hcorr  = "Corr_Obs"; hcorr  += std::min(obsNbr1,obsNbr2) ; hcorr += "_Obs";  hcorr  += std::max(obsNbr1, obsNbr2);
  for(size_t i=0; i<hObsCorr.size(); i++) {
    if(((TString)(hObsCorr[i]->GetName())) == hcorr) {
      if (obsNbr1 < obsNbr2) {
	hObsCorr[i] -> Fill(obsVal1,obsVal2, weight);
      } else {
	hObsCorr[i] -> Fill(obsVal2,obsVal1, weight);
      }
      return;
    }
  }
}


// member function to add observable values to the background histograms
void LRHelpFunctions::fillToBackgroundHists(const std::vector<double>& obsVals, double weight){
  for(size_t o=0; o<obsVals.size(); o++) hObsB[o]->Fill(obsVals[o], weight);
}

// member function to add observable values to the signal histograms
void LRHelpFunctions::fillToBackgroundHists(int obsNbr, double obsVal, double weight){
  TString obs = "Obs"; obs += obsNbr; obs += "_";
  for(size_t f = 0; f<hObsB.size(); f++){
    if(((TString)(hObsB[f]->GetName())).Contains(obs)) {
      hObsB[f]->Fill(obsVal, weight);
      return;
    }
  }
}

// member function to normalize the S and B spectra
void LRHelpFunctions::normalizeSandBhists(){
  for(size_t o=0; o<hObsS.size(); o++){
    // count entries in each histo. Do it this way instead of GetEntries method,
    // since the latter does not account for the weights.
    double nrSignEntries = 0, nrBackEntries = 0;
    for (int i=0; i<=hObsS[o]->GetNbinsX()+1; i++){
      nrSignEntries += hObsS[o]->GetBinContent(i);
      nrBackEntries += hObsB[o]->GetBinContent(i);
     }
    for(int b=0; b <= hObsS[o]->GetNbinsX()+1; b++){
      hObsS[o]->SetBinContent(b,hObsS[o]->GetBinContent(b)/(nrSignEntries));
      hObsB[o]->SetBinContent(b,hObsB[o]->GetBinContent(b)/(nrBackEntries));
      hObsS[o]->SetBinError(b,hObsS[o]->GetBinError(b)/(nrSignEntries));
      hObsB[o]->SetBinError(b,hObsB[o]->GetBinError(b)/(nrBackEntries));
    }
    //std::cout<<"Integral for obs"<<o<<" S: "<<hObsS[o]->Integral(0,10000)<<" & obs"<<o<<" B: "<<hObsB[o]->Integral(0,10000)<<std::endl;
  }
}

 
    
// member function to produce and fit the S/(S+B) histograms
void LRHelpFunctions::makeAndFitSoverSplusBHists(){
  for(size_t o=0; o<hObsS.size(); o++){
    for(int b=0; b <= hObsS[o]->GetNbinsX()+1; b++){
      if ((hObsS[o]->GetBinContent(b)+ hObsB[o]->GetBinContent(b)) > 0) {
        hObsSoverSplusB[o]->SetBinContent(b, hObsS[o]->GetBinContent(b) / (hObsS[o]->GetBinContent(b) + hObsB[o]->GetBinContent(b)));
        double error = sqrt( pow(hObsS[o]->GetBinError(b) * hObsB[o]->GetBinContent(b)/pow(hObsS[o]->GetBinContent(b) + hObsB[o]->GetBinContent(b),2),2)
	                   + pow(hObsB[o]->GetBinError(b) * hObsS[o]->GetBinContent(b)/pow(hObsS[o]->GetBinContent(b) + hObsB[o]->GetBinContent(b),2),2) );
        hObsSoverSplusB[o]->SetBinError(b,error);
      }
    }
    hObsSoverSplusB[o]->Fit(fObsSoverSplusB[o]->GetName(),"RQ");
  }
}


// member function to read the observable hists & fits from a root-file
void LRHelpFunctions::readObsHistsAndFits(const TString& fileName, const std::vector<int>& observables, bool readLRplots){
  hObsS.clear();
  hObsB.clear();
  hObsSoverSplusB.clear();
  fObsSoverSplusB.clear();
  TFile *fitFile = new TFile(fileName, "READ");
  if(observables[0] == -1){  
    std::cout<<" ... will read hists and fit for all available observables in file "<<fileName<<std::endl;
    TList *list = fitFile->GetListOfKeys();
    TIter next(list);
    TKey *el;
    while ((el = (TKey*)next())) {
      TString keyName = el->GetName();
      if(keyName.Contains("F_") && keyName.Contains("_SoverSplusB")){
        fObsSoverSplusB.push_back( new TF1(*((TF1*) el -> ReadObj())));
      }
      else if(keyName.Contains("_SoverSplusB")){
        hObsSoverSplusB.push_back( new TH1F(*((TH1F*) el -> ReadObj())));
      }
      else if(keyName.Contains("_S")){
        hObsS.push_back( new TH1F(*((TH1F*) el -> ReadObj())));
      }
      else if(keyName.Contains("_B")){
        hObsB.push_back( new TH1F(*((TH1F*) el -> ReadObj())));
      }
      else if(keyName.Contains("Corr")){
        hObsCorr.push_back( new TH2F(*((TH2F*) el -> ReadObj())));
      }
    }
  }
  else{
    obsNumbers = observables;
    for(unsigned int obs = 0; obs < observables.size(); obs++){
      std::cout<<"  ... will read hists and fit for obs "<<observables[obs]<<" from file "<<fileName<<std::endl;
      TString hStitle  = "Obs";   hStitle  += observables[obs];   hStitle += "_S";
      hObsS.push_back( new TH1F(*((TH1F*)fitFile->GetKey(hStitle)->ReadObj())));
      TString hBtitle  = "Obs";   hBtitle  += observables[obs];   hBtitle += "_B";
      hObsB.push_back( new TH1F(*((TH1F*)fitFile->GetKey(hBtitle)->ReadObj())));
      TString hSBtitle = "Obs";   hSBtitle += observables[obs];   hSBtitle += "_SoverSplusB";
      TString fSBtitle = "F_";  fSBtitle += hSBtitle;
      hObsSoverSplusB.push_back( new TH1F(*((TH1F*)fitFile->GetKey(hSBtitle)->ReadObj())));
      fObsSoverSplusB.push_back( new TF1(*((TF1*)fitFile->GetKey(fSBtitle)->ReadObj())));
      for(unsigned int obs2 = obs+1; obs2 < observables.size(); obs2++){
        TString hCorrtitle  = "Corr_Obs"; hCorrtitle  += observables[obs];   
                hCorrtitle += "_Obs";     hCorrtitle  += observables[obs2]; 
        hObsCorr.push_back( new TH2F(*((TH2F*)fitFile->GetKey(hCorrtitle)->ReadObj())));
      }
    }
  }
  
  if(readLRplots){
    constructPurity = true;
    std::cout<<"  ... will LR s and B histos from file "<<fileName<<std::endl;
    hLRtotS = new TH1F(*((TH1F*)fitFile->GetKey("hLRtotS")->ReadObj()));
    hLRtotB = new TH1F(*((TH1F*)fitFile->GetKey("hLRtotB")->ReadObj()));
    hLRtotSoverSplusB = new TH1F(*((TH1F*)fitFile->GetKey("hLRtotSoverSplusB")->ReadObj()));
    fLRtotSoverSplusB = new TF1(*((TF1*)fitFile->GetKey("fLRtotSoverSplusB") -> ReadObj()));
  }
}




// member function to store all observable plots and fits to a ROOT-file
void  LRHelpFunctions::storeToROOTfile(const TString& fname){
  TFile fOut(fname,"RECREATE");
  fOut.cd();
  for(size_t o=0; o<hObsS.size(); o++){
    hObsS[o] 		-> Write();
    hObsB[o] 		-> Write();
    hObsSoverSplusB[o] 	-> Write();
    fObsSoverSplusB[o] 	-> Write();
  }
  int hIndex = 0;
  for(size_t o=0; o<hObsS.size(); o++) {
    for(size_t o2=o+1; o2<hObsS.size(); o2++) {
      hObsCorr[hIndex] -> Write();
      ++ hIndex;
    }
  }
  if(constructPurity){
    hLRtotS 		-> Write();
    hLRtotB 		-> Write();
    hLRtotSoverSplusB 	-> Write();
    fLRtotSoverSplusB 	-> Write();
    hEffvsPur	        -> Write();
    hLRValvsPur	        -> Write();
    hLRValvsEff	        -> Write();
  }
  fOut.cd();
  fOut.Write();
  fOut.Close();
}





// member function to make some simple control plots and store them in a ps-file
void  LRHelpFunctions::storeControlPlots(const TString& fname){  
  TCanvas c("dummy","",1);
  c.Print(fname + "[","landscape");
  for(size_t o=0; o<hObsS.size(); o++) {
     TCanvas c2("c2","",1);
     c2.Divide(2,1);
     c2.cd(1);
     hObsS[o] -> SetLineColor(2);
     if(hObsB[o]->GetMaximum() > hObsS[o]->GetMaximum()){
       hObsB[o] -> Draw("hist");
       hObsS[o] -> Draw("histsame");
     }
     else
     {
       hObsS[o] -> Draw("hist");
       hObsB[o] -> Draw("histsame");
     }
     c2.cd(2);
     hObsSoverSplusB[o] -> Draw();
     fObsSoverSplusB[o] -> Draw("same");
     c2.Print(fname,"Landscape");
  }
  
  int hIndex = 0;
  for(size_t o=0; o<hObsS.size(); o++) {
    for(size_t o2=o+1; o2<hObsS.size(); o2++) {
      TCanvas cc("cc","",1);
      hObsCorr[hIndex] -> Draw("box");
      TPaveText pt(0.5,0.87,0.98,0.93,"blNDC");
      pt.SetFillStyle(1);
      pt.SetFillColor(0);
      pt.SetBorderSize(0);
      TString tcorr = "Corr. of "; tcorr += (int)(100.*hObsCorr[hIndex] -> GetCorrelationFactor()); tcorr += " %";
      //TText *text = pt.AddText(tcorr);
      pt.Draw("same");
      ++hIndex;
      cc.Print(fname,"Landscape");
    }
  }
  
  if(constructPurity){
    TCanvas c3("c3","",1);
    c3.Divide(2,1);
    c3.cd(1);
    hLRtotB -> Draw();
    hLRtotS -> SetLineColor(2);
    hLRtotS -> Draw("same");
    c3.cd(2);
    hLRtotSoverSplusB -> Draw();
    c3.Print(fname,"Landscape");
  
    TCanvas c4("c4","",1);
    hEffvsPur -> Draw("AL*");
    c4.Print(fname,"Landscape");

    TCanvas c5("c5","",1);
    hLRValvsPur -> Draw("AL*");
    c5.Print(fname,"Landscape");

    TCanvas c6("c6","",1);
    hLRValvsEff -> Draw("AL*");
    c6.Print(fname,"Landscape");
  } 
  c.Print(fname + "]","landscape");
}
   



// member function to calculate the LR value, using the S/N definition
double 	LRHelpFunctions::calcLRval(const std::vector<double>& vals){
  double logLR = 0.;
  for(size_t o=0; o<fObsSoverSplusB.size(); o++){
    double SoverSplusN = fObsSoverSplusB[o]->Eval(vals[o]);
    double SoverN = 1./((1./SoverSplusN) - 1.);
    logLR += log(SoverN);
  }  
  return logLR;
}


// member function to calculate the LR value, using the definition that was
// used in the P-TDR: S/(S+N)
 
double LRHelpFunctions::calcPtdrLRval(const std::vector<double>& vals, bool useCorrelation) {
  double logLR = 1.;
  for(size_t o=0; o<fObsSoverSplusB.size(); o++){
    double SoverSplusN = fObsSoverSplusB[o]->Eval(vals[o]);
    if (SoverSplusN<0.0001) SoverSplusN=0.0001;
    if (useCorrelation){
      double corr = 0;
      for(size_t j=0; j<fObsSoverSplusB.size(); j++){
        if (o==j) ++corr;
	else {
	  TString hcorr  = "Corr_Obs"; hcorr  += std::min(obsNumbers[o],obsNumbers[j]) ; hcorr += "_Obs";  hcorr  += std::max(obsNumbers[o],obsNumbers[j]);
	  for(size_t i=0; i<hObsCorr.size(); i++) {
	    if(((TString)(hObsCorr[i]->GetName())) == hcorr)
	      corr += fabs(hObsCorr[i]->GetCorrelationFactor());
	  }
	}
      }
     logLR *= pow(SoverSplusN/fObsSoverSplusB[o]->GetMaximum(), 1./corr);
    }
    else logLR *= SoverSplusN/fObsSoverSplusB[o]->GetMaximum();
  }
  //std::cout <<logLR<<std::endl;
  return logLR;
}


// member function to fill a signal contribution to the LR histogram
void  LRHelpFunctions::fillLRSignalHist(double val, double weight){
  //  std::cout<< "@@@===> LRHelpf Signal LRval = "<< val<<std::endl;
  hLRtotS -> Fill(val, weight);
}



// member function to fill a background contribution to the LR histogram
void  LRHelpFunctions::fillLRBackgroundHist(double val, double weight){
  // std::cout<< "@@@===> LRHelpf Backgroud LRval = "<< val<<std::endl;
  hLRtotB -> Fill(val, weight);
}
  
  
   
// member function to make and fit the purity vs. LRval histogram, and the purity vs. efficiency graph
void  LRHelpFunctions::makeAndFitPurityHists(){

  for(int b=0; b<=hLRtotS->GetNbinsX(); b++) {
    float Sint = hLRtotS->Integral(b, hLRtotS->GetNbinsX()+1);
    float Bint = hLRtotB->Integral(b, hLRtotB->GetNbinsX()+1);
    if (Sint + Bint > 0) {
      hLRtotSoverSplusB->SetBinContent(b,1. * Sint / (Sint + Bint));
      hLRtotSoverSplusB->SetBinError(b,sqrt((pow(Sint*sqrt(Bint),2)+pow(Bint*sqrt(Sint),2)))/pow((Sint+Bint),2));
    }
  }

  hLRtotS->GetXaxis()->SetTitle("Combined LR ratio");
  hLRtotB->GetXaxis()->SetTitle("Combined LR ratio");
  hLRtotSoverSplusB->GetXaxis()->SetTitle("Cut on Combined LR");
  hLRtotSoverSplusB->GetYaxis()->SetTitle("Purity");

  hLRtotSoverSplusB -> Fit(fLRtotSoverSplusB->GetName(),"RQ");  
  double totSignal = hLRtotS->Integral(0,hLRtotS->GetNbinsX()+1);
  double Eff[200], Pur[200], LRVal[200];
  if (hLRtotS->GetNbinsX()>200) {
    std::cout << "Number of bins of LR histograms can not execeed 200!"<<std::endl;
    return;
  }
  for(int cut=0; (cut<=hLRtotS->GetNbinsX())&&(cut<200) ; cut ++){
 	double LRcutVal = hLRtotS->GetBinLowEdge(cut);
	Eff[cut]   = hLRtotS->Integral(cut,hLRtotS->GetNbinsX()+1)/totSignal;
 	Pur[cut]   = fLRtotSoverSplusB->Eval(LRcutVal);
	LRVal[cut] = LRcutVal;
  }
  hEffvsPur = new TGraph(hLRtotS->GetNbinsX(),Eff,Pur);
  hEffvsPur -> SetName("hEffvsPur");
  hEffvsPur -> SetTitle("");
  hEffvsPur -> GetXaxis() -> SetTitle((TString)("Efficiency of cut on log combined LR"));
  hEffvsPur -> GetYaxis() -> SetTitle((TString)("Purity"));
  hEffvsPur -> GetYaxis() -> SetRangeUser(0,1.1);

  hLRValvsPur = new TGraph(hLRtotS->GetNbinsX(),LRVal,Pur);
  hLRValvsPur -> SetName("hLRValvsPur");
  hLRValvsPur -> SetTitle("");
  hLRValvsPur -> GetXaxis() -> SetTitle((TString)("Cut on the log combined LR value"));
  hLRValvsPur -> GetYaxis() -> SetTitle((TString)("Purity"));
  hLRValvsPur -> GetYaxis() -> SetRangeUser(0,1.1);

  hLRValvsEff = new TGraph(hLRtotS->GetNbinsX(),LRVal,Eff);
  hLRValvsEff -> SetName("hLRValvsEff");
  hLRValvsEff -> SetTitle("");
  hLRValvsEff -> GetXaxis() -> SetTitle((TString)("Cut on the log combined LR value"));
  hLRValvsEff -> GetYaxis() -> SetTitle((TString)("Efficiency of cut on log combined LR"));
  hLRValvsEff -> GetYaxis() -> SetRangeUser(0,1.1);

}   




// member function to calculate the probability for signal given a logLR value
double  LRHelpFunctions::calcProb(double logLR){
  return fLRtotSoverSplusB -> Eval(logLR);
}


// member to check if a certain S/S+B fit function is read from the root-file
bool 	LRHelpFunctions::obsFitIncluded(int o){
  bool included = false;
  TString obs = "_Obs"; obs += o; obs += "_";
  for(size_t f = 0; f<fObsSoverSplusB.size(); f++){    
    if(((TString)(fObsSoverSplusB[f]->GetName())).Contains(obs)) included = true;
  }
  return included;
}

void LRHelpFunctions::setXlabels(const std::vector<std::string> & xLabels)
{
  if (hObsS.size() != xLabels.size()) {
    std::cout << "LRHelpFunctions::setXlabels: Number of labels ("
         << xLabels.size()<< ") does not match number of obervables("
         << hObsS.size()<<").\n";
    return;
  }
  for(size_t i = 0; i<hObsS.size(); ++i){
    hObsS[i]->GetXaxis()->SetTitle(TString(xLabels[i]));
    hObsB[i]->GetXaxis()->SetTitle(TString(xLabels[i]));
    hObsSoverSplusB[i]->GetXaxis()->SetTitle(TString(xLabels[i]));
    hObsSoverSplusB[i]->GetYaxis()->SetTitle(TString("S/(S+B)"));
  }
}

void LRHelpFunctions::setYlabels(const std::vector<std::string> & yLabels)
{
  if (hObsS.size() != yLabels.size()) {
    std::cout << "LRHelpFunctions::setYlabels: Number of labels ("
	 << yLabels.size()<< ") does not match number of obervables("
	 << hObsS.size()<<").\n";
    return;
  }
  for(size_t i = 0; i<hObsS.size(); ++i){
    hObsS[i]->GetYaxis()->SetTitle(TString(yLabels[i]));
    hObsB[i]->GetYaxis()->SetTitle(TString(yLabels[i]));
  }
}

void  LRHelpFunctions::singlePlot(const TString& fname, int obsNbr, const TString& extension) {
  if (!obsFitIncluded(obsNbr)) return;

  TStyle *tdrStyle = gROOT->GetStyle("tdrStyle");
  tdrStyle->SetOptFit(0);
  tdrStyle->SetOptStat(0);
  tdrStyle->SetOptTitle(0);
//   tdrStyle->SetPadTopMargin(0.01);
//   tdrStyle->SetPadBottomMargin(0.01);
//   tdrStyle->SetPadLeftMargin(0.01);
//   tdrStyle->SetPadRightMargin(0.01);

  TCanvas c2("c2","",600,300);
  c2.SetTopMargin(0.01);
  c2.SetBottomMargin(0.01);
  c2.SetLeftMargin(0.01);
  c2.SetRightMargin(0.01);
  std::cout <<fname<<std::endl;
  c2.Divide(2,1);

  TString obs = "Obs"; obs += obsNbr; obs += "_";
  for(size_t o = 0; o<hObsB.size(); ++o){
    if(((TString)(hObsB[o]->GetName())).Contains(obs)) {
      c2.cd(1);
      hObsS[o] -> SetLineColor(2);
      if(hObsB[o]->GetMaximum() > hObsS[o]->GetMaximum()){
	hObsB[o] -> Draw("hist");
	hObsS[o] -> Draw("histsame");
      }
      else
      {
	hObsS[o] -> Draw("hist");
	hObsB[o] -> Draw("histsame");
      }
      c2.cd(2);

      hObsSoverSplusB[o] -> Draw();
      fObsSoverSplusB[o] -> Draw("same");
    }
  }
  std::cout << fname+"."+extension<<std::endl;
  c2.Print(fname+"."+extension);
}

void  LRHelpFunctions::purityPlot(const TString& fname,const TString& extension)
{
  TStyle *tdrStyle = gROOT->GetStyle("tdrStyle");
  tdrStyle->SetOptFit(0);
  tdrStyle->SetOptStat(0);
  tdrStyle->SetOptTitle(0);
//   tdrStyle->SetPadTopMargin(0.01);
//   tdrStyle->SetPadBottomMargin(0.01);
//   tdrStyle->SetPadLeftMargin(0.01);
//   tdrStyle->SetPadRightMargin(0.01);

  TCanvas c2("c2","",600,300);
  c2.SetTopMargin(0.01);
  c2.SetBottomMargin(0.01);
  c2.SetLeftMargin(0.01);
  c2.SetRightMargin(0.01);
  std::cout <<fname<<std::endl;
  c2.Divide(2,1);

  c2.cd(1);
  hLRValvsPur->Draw("AP");
  c2.cd(2);
  hEffvsPur->Draw("AP");
  c2.Print(fname+"Purity."+extension);

  hLRtotS->GetXaxis()->SetNdivisions(505);
  hLRtotB->GetXaxis()->SetNdivisions(505);
  TCanvas c3("c2","",300,300);
//   c3.SetTopMargin(0.01);
//   c3.SetBottomMargin(0.01);
//   c3.SetLeftMargin(0.01);
//   c3.SetRightMargin(0.01);
  hLRtotS->GetXaxis()->SetTitle("Combined LR");
  hLRtotB->GetXaxis()->SetTitle("Combined LR");

    hLRtotB -> Draw();
    hLRtotS -> SetLineColor(2);
    hLRtotS -> Draw("same");
  c3.Print(fname+"Dist."+extension);


    TCanvas c5("c3","",900,600);
    c5.Divide(2,2);
    c5.cd(1);
    hLRtotB -> Draw();
    hLRtotS -> SetLineColor(2);
    hLRtotS -> Draw("same");
    c5.cd(3);
    hLRValvsEff -> Draw("AP");
    c5.cd(2);
    hEffvsPur -> Draw("AP");
    c5.cd(4);
    hLRtotSoverSplusB -> Draw();
    c5.Print(fname+"all."+extension);
}
