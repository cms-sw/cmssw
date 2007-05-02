#include "TopQuarkAnalysis/TopJetCombination/interface/LRHelpFunctions.h"




//////////////////////////////////////////////////////////////////////////////////////////////
//
// Function to calculate the S over B for all the bins and fit the resulting distribution
//////////////////////////////////////////////////////////////////////////////////////////////

TH1F SoverB(TH1F *hsign, TH1F *hback, int obsNr) {
  TString ht = "hSoverB_Obs"; ht += obsNr;
  TH1F hSoverB(ht,ht,hsign->GetNbinsX(),hsign->GetXaxis()->GetXmin(),hsign->GetXaxis()->GetXmax());
  TString t = "Obs"; t += obsNr; hSoverB.GetXaxis()->SetTitle(t);
  hSoverB.GetYaxis()->SetTitle((TString)("S/B"));
  for (int k = 1; k <= hsign->GetNbinsX(); k++) {
    if (hsign->GetBinContent(k) > 0 && hback->GetBinContent(k) > 0) {
      hSoverB.SetBinContent(k, hsign->GetBinContent(k) / hback->GetBinContent(k));
      double error = sqrt((pow(hsign->GetBinError(k)/hback->GetBinContent(k),2)+pow(hsign->GetBinContent(k)*hback->GetBinError(k)/pow(hback->GetBinContent(k),2),2)));
      hSoverB.SetBinError(k,error);
    }
  }
  return hSoverB;
}






////////////////////////////////////////////////////////////////////////////////
//
// Function to make the Purity of good combinations vs. LRtot-plot
////////////////////////////////////////////////////////////////////////////////

TH1F makePurityPlot(TH1F *hLRtotS, TH1F *hLRtotB){
  TH1F hPurity("hPurity","",hLRtotS->GetNbinsX(),hLRtotS->GetXaxis()->GetXmin(),hLRtotS->GetXaxis()->GetXmax());
  hPurity.GetXaxis()->SetTitle((TString)("log combined LR"));
  hPurity.GetYaxis()->SetTitle((TString)("Purity"));
  for(int b=0; b<=hLRtotS->GetNbinsX(); b++) {
    if(!(hLRtotS->GetBinContent(b)==0 && hLRtotB->GetBinContent(b)==0)) {
      hPurity.SetBinContent(b,hLRtotS->GetBinContent(b)/(hLRtotS->GetBinContent(b)+hLRtotB->GetBinContent(b)));
      Float_t error = sqrt((pow(hLRtotS->GetBinContent(b)*hLRtotB->GetBinError(b),2)+pow(hLRtotB->GetBinContent(b)*hLRtotS->GetBinError(b),2)))/
      				pow((hLRtotS->GetBinContent(b)+hLRtotB->GetBinContent(b)),2);
      hPurity.SetBinError(b,error);
    }
  }
  return hPurity;
}











///////////////////////////////////////////////////////////////////////////////////////
//
// Function to make the Purity of correct combinations vs. Efficiency of a LRtot-cut
///////////////////////////////////////////////////////////////////////////////////////

TGraph makeEffVsPurGraph(TH1F *hLRtotS, TF1 *fPurity){
  double Eff[100], Pur[100];
  for(int cut=0; cut<=hLRtotS->GetNbinsX(); cut ++){
 	Double_t LRcutVal = hLRtotS->GetBinCenter(cut+1);
	Eff[cut] = 1.- hLRtotS->Integral(0,cut+1)/hLRtotS->Integral(0,50);
	Pur[cut] = fPurity->Eval(LRcutVal);
  }
  TGraph hEffvsPur(hLRtotS->GetNbinsX(),Eff,Pur);
  hEffvsPur.SetName("hEffvsPur");
  hEffvsPur.GetXaxis()->SetTitle((TString)("efficiency of cut on log combined LR"));
  hEffvsPur.GetYaxis()->SetTitle((TString)("Purity"));
  return hEffvsPur;
}
