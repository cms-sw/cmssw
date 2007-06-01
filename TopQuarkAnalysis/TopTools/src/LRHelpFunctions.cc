//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: LRHelpFunctions.cc,v 1.1 2007/05/24 13:19:47 heyninck Exp $
//
#include "TopQuarkAnalysis/TopTools/interface/LRHelpFunctions.h"
#include "TopQuarkAnalysis/TopEventProducers/bin/tdrstyle.C"

// constructors
LRHelpFunctions::LRHelpFunctions() {}
LRHelpFunctions::LRHelpFunctions(vector<int> obsNr, int nrBins, vector<double> obsMin, vector<double> obsMax,
                                 vector<const char*> functions, int nrLRbins, double LRmin, double LRmax, const char* LRfunction) { 
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

  // create LR histograms
  hLRtotS = new TH1F("hLRtotS","hLRtotS",nrLRbins,LRmin,LRmax);
  hLRtotB = new TH1F("hLRtotB","hLRtotB",nrLRbins,LRmin,LRmax);
  hLRtotSoverSplusB = new TH1F("hLRtotSoverSplusB","hLRtotSoverSplusB",nrLRbins,LRmin,LRmax);
    
  // create LR fit function
  fLRtotSoverSplusB = new TF1("fLRtotSoverSplusB",LRfunction,LRmin,LRmax);
}


// destructor
LRHelpFunctions::~LRHelpFunctions() {}



// member function to add observable values to the signal histograms
void LRHelpFunctions::fillToSignalHists(vector<double> obsVals){
  int hIndex = 0;
  for(size_t o=0; o<obsVals.size(); o++) {
    hObsS[o]->Fill(obsVals[o]);
    for(size_t o2=o+1; o2<obsVals.size(); o2++) {
      hObsCorr[hIndex] -> Fill(obsVals[o],obsVals[o2]);
      ++hIndex;
    }
  }
}
  
  
    
// member function to add observable values to the background histograms
void LRHelpFunctions::fillToBackgroundHists(vector<double> obsVals){
  for(size_t o=0; o<obsVals.size(); o++) hObsB[o]->Fill(obsVals[o]);
}
 
    
// member function to produce and fit the S/(S+B) histograms
void LRHelpFunctions::makeAndFitSoverSplusBHists(){
  for(size_t o=0; o<hObsS.size(); o++){
    for(int b=0; b <= hObsS[o]->GetNbinsX(); b++){
      if (hObsS[o]->GetBinContent(b) > 0 && hObsB[o]->GetBinContent(b) > 0) {
        hObsSoverSplusB[o]->SetBinContent(b, hObsS[o]->GetBinContent(b) / hObsB[o]->GetBinContent(b));
        double error = sqrt((pow(hObsS[o]->GetBinError(b)/hObsB[o]->GetBinContent(b),2)+pow(hObsS[o]->GetBinContent(b)*hObsB[o]->GetBinError(b)/pow(hObsB[o]->GetBinContent(b),2),2)));
        hObsSoverSplusB[o]->SetBinError(b,error);
      }
    }
    hObsSoverSplusB[o]->Fit(fObsSoverSplusB[o]->GetName(),"RQ");
  }
}


// member function to read the observable hists & fits from a root-file
void LRHelpFunctions::readObsHistsAndFits(TString fileName, bool readLRplots){
  hObsS.clear();
  hObsB.clear();
  hObsSoverSplusB.clear();
  fObsSoverSplusB.clear();
  TFile *fitFile = new TFile(fileName, "READ");
  TList *list = fitFile->GetListOfKeys();
  TIter next(list);
  TKey *el;
  while ((el = (TKey*)next())) {
    TString keyName = el->GetName();
    if(keyName.Contains("SoverSplusB") && keyName.Contains("Obs")) {
      TH1F tmp =  *((TH1F*) el -> ReadObj());
      hObsSoverSplusB.push_back( new TH1F(tmp));
      TString ft = "F_"; ft += keyName;
      fObsSoverSplusB.push_back( new TF1(*(((TH1F*) el -> ReadObj()) -> GetFunction(ft))) );
      keyName.Remove(keyName.Index("_"),keyName.Length());
      keyName.Remove(0,3);
      selObs.push_back(keyName.Atoi());
    }
    else if(keyName.Contains("S") && keyName.Contains("Obs")){
      TH1F tmp =  *((TH1F*) el -> ReadObj());
      hObsS.push_back( new TH1F(tmp));
    }
    else if(keyName.Contains("B") && keyName.Contains("Obs")){
      TH1F tmp =  *((TH1F*) el -> ReadObj());
      hObsB.push_back( new TH1F(tmp));
    }
  }
  
  if(readLRplots){
    hLRtotS = new TH1F(*((TH1F*)fitFile->GetKey("hLRtotS")->ReadObj()));
    hLRtotB = new TH1F(*((TH1F*)fitFile->GetKey("hLRtotB")->ReadObj()));
    hLRtotSoverSplusB = new TH1F(*((TH1F*)fitFile->GetKey("hLRtotSoverSplusB")->ReadObj()));
    fLRtotSoverSplusB = new TF1(*((TF1*)(((TH1F*)fitFile->GetKey("hLRtotSoverSplusB")->ReadObj())->GetFunction("fLRtotSoverSplusB"))));
  }
}




// member function to store all observable plots and fits to a ROOT-file
void  LRHelpFunctions::storeToROOTfile(TString fname){
  TFile fOut(fname,"RECREATE");
  fOut.cd();
  for(size_t o=0; o<hObsS.size(); o++){
    hObsS[o] 		-> Write();
    hObsB[o] 		-> Write();
    hObsSoverSplusB[o] 	-> Write();
  }
  for(size_t o=0; o<hObsS.size(); o++) {
    int hIndex = 0;
    for(size_t o2=o+1; o2<hObsS.size(); o2++) {
      hObsCorr[hIndex] -> Write();
      ++ hIndex;
    }
  }
  hLRtotS 		-> Write();
  hLRtotB 		-> Write();
  hLRtotSoverSplusB 	-> Write();
  fOut.cd();
  fOut.Write();
  fOut.Close();
}


// member function to make some simple control plots and store them in a ps-file
void  LRHelpFunctions::storeControlPlots(TString fname){  
  TCanvas c("dummy","",1);
  c.Print(fname + "[","landscape");
  for(size_t o=0; o<hObsS.size(); o++) {
     TCanvas c2("c2","",1);
     c2.Divide(2,1);
     c2.cd(1);
     hObsB[o] -> Draw();
     hObsS[o] -> SetLineColor(2);
     hObsS[o] -> Draw("same");
     c2.cd(2);
     hObsSoverSplusB[o] -> Draw();
     fObsSoverSplusB[o] -> Draw("same");
     c2.Print(fname,"Landscape");
  }
  
  for(size_t o=0; o<hObsS.size(); o++) {
    int hIndex = 0;
    for(size_t o2=o+1; o2<hObsS.size(); o2++) {
      TCanvas cc("cc","",1);
      hObsCorr[hIndex] -> Draw("box");
      TPaveText pt(0.5,0.87,0.98,0.93,"blNDC");
      pt.SetFillStyle(1);
      pt.SetFillColor(0);
      pt.SetBorderSize(0);
      TString tcorr = "Corr. of "; tcorr += (int)(100.*hObsCorr[hIndex] -> GetCorrelationFactor()); tcorr += " %";
      TText *text = pt.AddText(tcorr);
      pt.Draw("same");
      ++hIndex;
      cc.Print(fname,"Landscape");
    }
  }
  
  TCanvas c3("c3","",1);
  c3.Divide(2,1);
  c3.cd(1);
  hLRtotB -> Draw();
  hLRtotS -> SetLineColor(2);
  hLRtotS -> Draw("same");
  c3.cd(2);
  hLRtotSoverSplusB -> Draw();
  c3.Print(fname,"Landscape");
  
  c.Print(fname + "]","landscape");
}
   



// member function to fill a signal contribution to the LR histogram 
double 	LRHelpFunctions::calcLRval(vector<double> vals){
  double logLR = 0.;
  for(size_t o=0; o<fObsSoverSplusB.size(); o++){
    double SoverSplusN = fObsSoverSplusB[o]->Eval(vals[o]);
    double SoverN = 1./((1./SoverSplusN) - 1.);
    logLR += log(SoverN);
  }  
  return logLR;
}
 
 
   
// member function to fill a signal contribution to the LR histogram 
void  LRHelpFunctions::fillLRSignalHist(double val){
  hLRtotS -> Fill(val);
}   
  
  
   
// member function to fill a background contribution to the LR histogram 
void  LRHelpFunctions::fillLRBackgroundHist(double val){
  hLRtotB -> Fill(val);
}    
  
  
   
// member function to make and fit the purity vs. LRval histogram, and the purity vs. efficiency graph
void  LRHelpFunctions::makeAndFitPurityHists(){
  for(int b=0; b<=hLRtotS->GetNbinsX(); b++) {
    if(!(hLRtotS->GetBinContent(b)==0 && hLRtotB->GetBinContent(b)==0)) {
      hLRtotSoverSplusB->SetBinContent(b,hLRtotS->GetBinContent(b)/(hLRtotS->GetBinContent(b)+hLRtotB->GetBinContent(b)));
      Float_t error = sqrt((pow(hLRtotS->GetBinContent(b)*hLRtotB->GetBinError(b),2)+pow(hLRtotB->GetBinContent(b)*hLRtotS->GetBinError(b),2)))/
      				pow((hLRtotS->GetBinContent(b)+hLRtotB->GetBinContent(b)),2);
      hLRtotSoverSplusB->SetBinError(b,error);
    }
  }
  hLRtotSoverSplusB -> Fit(fLRtotSoverSplusB->GetName(),"RQ");
}   



// member function to calculate the probability for signal given a logLR value
double  LRHelpFunctions::calcProb(double logLR){
  return fLRtotSoverSplusB -> Eval(logLR);
}


// member function to check if a certain observable was used
bool  LRHelpFunctions::isIncluded(int obs){
  bool found = false;
  for(size_t o=0; o<selObs.size(); o++){
    if(selObs[o] == obs) found = true;
  }
  return found;
}
