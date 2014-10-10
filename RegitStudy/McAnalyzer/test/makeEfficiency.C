#if !defined(__CINT__) || defined(__MAKECINT__)
#include <Riostream.h>
#include <TROOT.h>
#include <TDirectory.h>
#include <TSystem.h>

#include <TF1.h>
#include <TFile.h>
#include <TGraphAsymmErrors.h>
#include <TH1.h>
#include <TH2D.h>

#include <TCanvas.h>
#include "TChain.h"
#include <TCut.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TMath.h>
#include <TPaletteAxis.h>
#include <TStyle.h>

// miscellaneous  
#include <fstream>
#include <map>
#include <string>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#endif
using namespace std;


//____________________________________________________________________________________
static bool smuacc(double pt, double eta)
{
   if (fabs(eta)<1.3 && pt>3.3) return true;
   else if (fabs(eta)>=1.3 && fabs(eta)<2.2 && (pt/sin(2*atan(exp(-eta))))>2.9) return true;
   else if (fabs(eta)>=2.2 && fabs(eta)<2.4 && pt>0.8) return true;
   else return false;
}
//____________________________________________________________________________________
TGraphAsymmErrors *calcEfficiency(TH1* h1, TH1* h2)
{

  TH1 *phUp   = (TH1 *)h2->Clone("phUp");
  TH1 *phDown = (TH1 *)h1->Clone("phDown");
  
  phUp->SetDirectory(0);
  phDown->SetDirectory(0);
  
  TGraphAsymmErrors *pgEfficiency = new TGraphAsymmErrors();
  pgEfficiency->BayesDivide(phUp,phDown,"");
  return pgEfficiency;
}
// const char* infilePairStd="files/tuple_jpsi_v11_minbias_ptSignal03_pnSimRecoPairType1.root",
// 		    const char* infilePairRegit="files/tuple_jpsi_v11_minbias_ptSignal03_pnSimRecoPairType2.root",
// 		    const char* infileSingleStd="files/tuple_jpsi_v11_minbias_ptSignal03_pnSimRecoTrkType1.root",
// 		    const char* infileSingleRegit="files/tuple_jpsi_v11_minbias_ptSignal03_pnSimRecoTrkType2.root",

// const char* infilePairStd="/tmp/camelia/tuple_jpsi_v11_minbias_ptSignal03_pnSimRecoPairType1.root",
// 		    const char* infilePairRegit="/tmp/camelia/tuple_jpsi_v11_minbias_ptSignal03_pnSimRecoPairType2.root",
// 		    const char* infileTrkStd="/tmp/camelia/tuple_jpsi_v11_minbias_ptSignal03_pnSimRecoTrkType1.root",
// 		    const char* infileTrkRegit="/tmp/camelia/tuple_jpsi_v11_minbias_ptSignal03_pnSimRecoTrkType2.root",
//____________________________________________________________________________________
void makeEfficiency(const char* pTuplePairStd="mcmatchanalysis/pnSimRecoPairType1",
		    const char* pTuplePairRegit="mcmatchanalysis/pnSimRecoPairType2",
		    const char* pTupleTrkStd="mcmatchanalysis/pnSimRecoTrkType1",
		    const char* pTupleTrkRegit="mcmatchanalysis/pnSimRecoTrkType2",
		    const char* pOutFilePath="outputfigs",
		    const char* pOutFileName="effs_bjpsi",
		    Bool_t bSavePlots=true)
{
  gROOT->Macro("/afs/cern.ch/user/m/mironov/utilities/setStyle.C+");
  gStyle->SetPalette(1);
  const char* aInFileLocation[4] = {"",
				    // "/tmp/camelia/tuple_jpsi_v11_minbias"
				    "/afs/cern.ch/user/e/echapon/workspace/private/jpsi_PbPb_5_3_17/CMSSW_5_3_17/src/RegitStudy/McAnalyzer/test/ntuples_bjpsi/ntuples_pp",
				    "/afs/cern.ch/user/e/echapon/workspace/private/jpsi_PbPb_5_3_17/CMSSW_5_3_17/src/RegitStudy/McAnalyzer/test/ntuples_bjpsi/ntuples_regit",
				    "/afs/cern.ch/user/e/echapon/workspace/private/jpsi_PbPb_5_3_17/CMSSW_5_3_17/src/RegitStudy/McAnalyzer/test/ntuples_bjpsi/ntuples_regit_3LastRegitStepsAsPp"
				   };
  const int nPtBins = 2;
  const char* aPtSignalBins[nPtBins] = {"",
               ""
					//	"_ptSignal03_",
					//	"_ptSignal36_",
               // "_ptSignal69_",
					//"_ptSignal1215_",
					//	"_ptSignal1530_"
  };
  bool bDoSinglePlots =  true;
  bool bDo2D          = true;
  bool bDoPairPlots   =  true;
  float ptPairMin = 0;
  float ptPairMax = 20;
  int nPtPairBins = 10;

  float ptTrkMin = 0;
  float ptTrkMax = 20;
  int nPtTrkBins = 10;

  TCut pairGenSelection("smuacc(pt1,eta1)&&smuacc(pt2,eta2) &&npixelhits1>2&&npixelhits2>2 &&nmuonhits1>0&&nmuonhits2>0");
  TCut pairRecoSelection("smuacc(ptreco1,etareco1)&&smuacc(ptreco2,etareco2) &&nvalidpixelhitsreco1>2&&nvalidpixelhitsreco2>2 &&nvalidmuonhitsreco1>0&&nvalidmuonhitsreco2>0 &&minvreco>2.6&&minvreco<3.5&&chi2ndofreco1<4&&chi2ndofreco2<4");

  TCut trkGenSelection("TMath::Abs(eta)<2.4&&(abs(idparent)<550&&abs(idparent)>442) &&npixelhits>2&&nmuonhits>0");
  TCut trkRecoSelection("TMath::Abs(eta)<2.4&&(abs(idparent)<550&&abs(idparent)>442) &&nvalidpixelhitsreco>2&&nvalidmuonhitsreco>0 &&nmatch>0&&chi2ndofreco<4");

  // TCut pairGenSelection("TMath::Abs(eta1)<2.4&&TMath::Abs(eta2)<2.4&&npixelhits1>1&&npixelhits2>1&&nmuonhits1>0&&nmuonhits2>0");
//   TCut pairRecoSelection("nmuonhits1>0&&nmuonhits2>0&&minvreco>2.6&&minvreco<3.5");
//   TCut trkGenSelection("TMath::Abs(eta)<2.4&&idparent==443&&npixelhits>1&&nmuonhits>0");
//   TCut trkRecoSelection("nvalidmuonhitsreco>0");
  
  // axis pair
  TH1F *phPtTmp = new TH1F("phPtTmp",";p_{T}^{#mu#mu}[GeV/c];Efficiency",1,ptPairMin,ptPairMax);
  TH1F *phYTmp  = new TH1F("phYTmp",";y^{#mu#mu};Efficiency",1,-2.4,2.4);
  phPtTmp->SetMinimum(0.005);
  phYTmp->SetMinimum(0.005);
  phPtTmp->SetMaximum(1.);
  phYTmp->SetMaximum(1.);
  TH2F *phPtYTmp = new TH2F("phPtYTmp",";y;p_{T}^{#mu#mu}[GeV/c]",1,-2.4,2.4,1,ptPairMin,ptPairMax);
 

  // axis single
  TH1F *phPtTmp2 = new TH1F("phPtTmp2",";p_{T}^{#mu}[GeV/c];Efficiency",1,ptTrkMin,ptTrkMax);
  phPtTmp2->SetMinimum(0.005);
  phPtTmp2->SetMaximum(1.);
    
  TH1F *phEtaTmp  = new TH1F("phEtaTmp",";#eta^{#mu};Efficiency",1,-2.4,2.4);
  phEtaTmp->SetMinimum(0.005);
  phEtaTmp->SetMaximum(1.);
  TH2F *phPtEtaTmp = new TH2F("phPtEtaTmp",";#eta;p_{T}^{#mu}[GeV/c]",1,-2.4,2.4,1,ptTrkMin,ptTrkMax);
  // TH2F *phPtEtaTmp = new TH2F("phPtEtaTmp",";#eta;p_{T}^{#mu}[GeV/c]",1,-2.4,2.4,1,ptTrkMin,5);

  if(bDoPairPlots)
    {
      TChain *ptPairPp   = new TChain(Form("%s",pTuplePairStd));
      TChain *ptPairStd   = new TChain(Form("%s",pTuplePairStd));
      TChain *ptPairRegit = new TChain(Form("%s",pTuplePairRegit));
      for(int iptbin=1; iptbin<nPtBins; iptbin++)
	{
	  const char* infilePairPp   = Form("%s%s.root",aInFileLocation[1],aPtSignalBins[iptbin]);
	  ptPairPp->Add(infilePairPp);
	  const char* infilePairStd = Form("%s%s.root",aInFileLocation[2],aPtSignalBins[iptbin]);
	  ptPairStd->Add(infilePairStd);
	  const char* infilePairRegit = Form("%s%s.root",aInFileLocation[3],aPtSignalBins[iptbin]);
	  ptPairRegit->Add(infilePairRegit);
	}
   
  
      // ########## pair plots!!!
      TH1D *phPtGenRecoPair_pp = new TH1D("phPtGenRecoPair_pp","phPtGenRecoPair_pp",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *phPtGenPair_pp     = new TH1D("phPtGenPair_pp","phPtGenPair_pp",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *phYGenRecoPair_pp  = new TH1D("phYGenRecoPair_pp","phYGenRecoPair_pp",12,-2.4,2.4);
      TH1D *phYGenPair_pp      = new TH1D("phYGenPair_pp","phYGenPair_pp",12,-2.4,2.4);

      phPtGenRecoPair_pp->Sumw2();
      phPtGenPair_pp->Sumw2();
      phYGenRecoPair_pp->Sumw2();
      phYGenPair_pp->Sumw2();
      
      TH1D *phPtGenRecoPair_std = new TH1D("phPtGenRecoPair_std","phPtGenRecoPair_std",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *phPtGenPair_std     = new TH1D("phPtGenPair_std","phPtGenPair_std",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *phYGenRecoPair_std  = new TH1D("phYGenRecoPair_std","phYGenRecoPair_std",12,-2.4,2.4);
      TH1D *phYGenPair_std      = new TH1D("phYGenPair_std","phYGenPair_std",12,-2.4,2.4);

      phPtGenRecoPair_std->Sumw2();
      phPtGenPair_std->Sumw2();
      phYGenRecoPair_std->Sumw2();
      phYGenPair_std->Sumw2();
      
      TH1D *phPtGenRecoPair_regit = new TH1D("phPtGenRecoPair_regit","phPtGenRecoPair_regit",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *phPtGenPair_regit     = new TH1D("phPtGenPair_regit","phPtGenPair_regit",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *phYGenRecoPair_regit  = new TH1D("phYGenRecoPair_regit","phYGenRecoPair_regit",12,-2.4,2.4);
      TH1D *phYGenPair_regit      = new TH1D("phYGenPair_regit","phYGenPair_regit",12,-2.4,2.4);
      phPtGenRecoPair_regit->Sumw2();
      phPtGenPair_regit->Sumw2();
      phYGenRecoPair_regit->Sumw2();
      phYGenPair_regit->Sumw2();
      
      // 2d histos
      TH2F *phPtYGenRecoPair_pp  = new TH2F("phPtYGenRecoPair_pp",";y;p_{T}^{#mu#mu}[GeV/c]",48,-2.4,2.4,nPtPairBins,ptPairMin,ptPairMax);
      TH2F *phPtYGenPair_pp      = new TH2F("phPtYGenPair_pp",";y;p_{T}^{#mu#mu}[GeV/c]",48,-2.4,2.4,nPtPairBins,ptPairMin,ptPairMax);
      TH2F *phPtYGenRecoPair_std  = new TH2F("phPtYGenRecoPair_std",";y;p_{T}^{#mu#mu}[GeV/c]",48,-2.4,2.4,nPtPairBins,ptPairMin,ptPairMax);
      TH2F *phPtYGenPair_std      = new TH2F("phPtYGenPair_std",";y;p_{T}^{#mu#mu}[GeV/c]",48,-2.4,2.4,nPtPairBins,ptPairMin,ptPairMax);
      TH2F *phPtYGenRecoPair_regit = new TH2F("phPtYGenRecoPair_regit",";y;p_{T}^{#mu#mu}[GeV/c]",48,-2.4,2.4,nPtPairBins,ptPairMin,ptPairMax);
      TH2F *phPtYGenPair_regit     = new TH2F("phPtYGenPair_regit",";y;p_{T}^{#mu#mu}[GeV/c]",48,-2.4,2.4,nPtPairBins,ptPairMin,ptPairMax);
      
      TCanvas *pcTemp = new TCanvas("pcTemp","pcTemp", 900, 400);
      pcTemp->Divide(3,1);
      pcTemp->cd(1);
      ptPairPp->Draw("pt>>phPtGenPair_pp","","E");
      ptPairPp->Draw("pt>>phPtGenPair_pp",pairGenSelection,"E");
      ptPairPp->Draw("ptreco>>phPtGenRecoPair_pp",pairRecoSelection,"Esame");
   
      pcTemp->cd(2);
      ptPairPp->Draw("y>>phYGenPair_pp",pairGenSelection,"E");
      ptPairPp->Draw("yreco>>phYGenRecoPair_pp",pairRecoSelection,"Esame");
      
      pcTemp->cd(1);
      ptPairStd->Draw("pt>>phPtGenPair_std",pairGenSelection,"Esame");
      ptPairStd->Draw("ptreco>>phPtGenRecoPair_std",pairRecoSelection,"Esame");
   
      pcTemp->cd(2);
      ptPairStd->Draw("y>>phYGenPair_std",pairGenSelection,"Esame");
      ptPairStd->Draw("yreco>>phYGenRecoPair_std",pairRecoSelection,"Esame");
      
      pcTemp->cd(1);
      ptPairRegit->Draw("pt>>phPtGenPair_regit",pairGenSelection,"Esame");
      ptPairRegit->Draw("ptreco>>phPtGenRecoPair_regit",pairRecoSelection,"Esame");
      pcTemp->cd(2);
      ptPairRegit->Draw("y>>phYGenPair_regit",pairGenSelection,"Esame");
      ptPairRegit->Draw("yreco>>phYGenRecoPair_regit",pairRecoSelection,"Esame");
      
      ptPairPp->Draw("pt:y>>phPtYGenPair_pp",pairGenSelection,"colz");
      ptPairPp->Draw("ptreco:yreco>>phPtYGenRecoPair_pp",pairRecoSelection,"colz");
      ptPairStd->Draw("pt:y>>phPtYGenPair_std",pairGenSelection,"colz");
      ptPairStd->Draw("ptreco:yreco>>phPtYGenRecoPair_std",pairRecoSelection,"colz");
      ptPairRegit->Draw("pt:y>>phPtYGenPair_regit",pairGenSelection,"colz");
      ptPairRegit->Draw("ptreco:yreco>>phPtYGenRecoPair_regit",pairRecoSelection,"colz");
      
      TH1D *pgPtEff_pp = new TH1D("pgPtEff_pp","pgPtEff_pp",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *pgYEff_pp  = new TH1D("pgYEff_pp","pgYEff_pp",12,-2.4,2.4);
      pgPtEff_pp->Sumw2();
      pgYEff_pp->Sumw2();
      TH1D *pgPtEff_std = new TH1D("pgPtEff_std","pgPtEff_std",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *pgYEff_std  = new TH1D("pgYEff_std","pgYEff_std",12,-2.4,2.4);
      pgPtEff_std->Sumw2();
      pgYEff_std->Sumw2();
      TH1D *pgPtEff_regit =  new TH1D("pgPtEff_regit","pgPtEff_regit",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *pgYEff_regit  =  new TH1D("pgYEff_regit","pgYEff_regit",12,-2.4,2.4);
      pgPtEff_regit->Sumw2();
      pgYEff_regit->Sumw2();

      pgPtEff_pp->Divide(phPtGenRecoPair_pp,phPtGenPair_pp,1,1,"b");
      pgYEff_pp->Divide(phYGenRecoPair_pp,phYGenPair_pp,1,1,"b");
      pgPtEff_std->Divide(phPtGenRecoPair_std,phPtGenPair_std,1,1,"b");
      pgYEff_std->Divide(phYGenRecoPair_std,phYGenPair_std,1,1,"b");
      pgPtEff_regit->Divide(phPtGenRecoPair_regit,phPtGenPair_regit,1,1,"b");
      pgYEff_regit->Divide(phYGenRecoPair_regit,phYGenPair_regit,1,1,"b");

      // 2d histos
      TH2D *pgPtYEff_pp   = new TH2D("pgPtYEff_pp","pgPtYEff_pp",48,-2.4,2.4,nPtPairBins,ptPairMin,ptPairMax);
      TH2D *pgPtYEff_std   = new TH2D("pgPtYEff_std","pgPtYEff_std",48,-2.4,2.4,nPtPairBins,ptPairMin,ptPairMax);
      TH2D *pgPtYEff_regit = new TH2D("pgPtYEff_regit","pgPtYEff_regit",48,-2.4,2.4,nPtPairBins,ptPairMin,ptPairMax);
      pgPtYEff_pp->Divide(phPtYGenRecoPair_pp,phPtYGenPair_pp,1,1,"b");
      pgPtYEff_std->Divide(phPtYGenRecoPair_std,phPtYGenPair_std,1,1,"b");
      pgPtYEff_regit->Divide(phPtYGenRecoPair_regit,phPtYGenPair_regit,1,1,"b");

      // drawing
      pgPtEff_pp->SetLineColor(3);
      pgPtEff_pp->SetMarkerColor(3);
      pgPtEff_pp->SetMarkerStyle(3);
      
      pgYEff_pp->SetLineColor(3);
      pgYEff_pp->SetMarkerColor(3);
      pgYEff_pp->SetMarkerStyle(3);
      
      pgPtEff_std->SetLineColor(4);
      pgPtEff_std->SetMarkerColor(4);
      pgPtEff_std->SetMarkerStyle(4);
      
      pgYEff_std->SetLineColor(4);
      pgYEff_std->SetMarkerColor(4);
      pgYEff_std->SetMarkerStyle(4);
      
      pgPtEff_regit->SetLineColor(2);
      pgPtEff_regit->SetMarkerColor(2);
      pgPtEff_regit->SetMarkerStyle(25);
      
      pgYEff_regit->SetLineColor(2);
      pgYEff_regit->SetMarkerColor(2);
      pgYEff_regit->SetMarkerStyle(25);
      
      TCanvas *pcPairEff = new TCanvas("pcPairEff","pcPairEff",900,500);
      pcPairEff->Divide(2,1);
      pcPairEff->cd(1);// gPad->SetLogy();
      phPtTmp->Draw();
      pgPtEff_pp->Draw("pl same");
      pgPtEff_std->Draw("pl same");
      pgPtEff_regit->Draw("pl same");
      
      pcPairEff->cd(2); //gPad->SetLogy();
      phYTmp->Draw();
      pgYEff_pp->Draw("pl same");
      pgYEff_std->Draw("pl same");
      pgYEff_regit->Draw("pl same");
      
      TLegend *t = new TLegend(0.34,0.78,0.94,0.91);
      t->SetBorderSize(0);
      t->SetFillStyle(0);
      t->AddEntry(pgPtEff_pp,"Pp_reco","pl");
      t->AddEntry(pgPtEff_std,"Std_reco","pl");
      t->AddEntry(pgPtEff_regit,"Regit_reco","pl");
      t->Draw();

      //2D histo
      if(bDo2D)
	{
	  TLatex lx;
	  TCanvas *pcPtYPairEff = new TCanvas("pcPtYPairEff","pcPtYPairEff",1500,600);
	  pcPtYPairEff->Divide(3,1);
	  pcPtYPairEff->cd(1);      
	  pcPtYPairEff->GetPad(1)->SetLeftMargin(0.16);
	  pcPtYPairEff->GetPad(1)->SetRightMargin(0.15);
	  phPtYTmp->Draw();
	  pgPtYEff_pp->GetZaxis()->SetRangeUser(0.0,1.0);
	  pgPtYEff_pp->Draw("colz same");
	  lx.DrawLatex(0.,2.,"Pp_reco");
	  gPad->Update();

	  pcPtYPairEff->cd(2);      
	  pcPtYPairEff->GetPad(2)->SetLeftMargin(0.16);
	  pcPtYPairEff->GetPad(2)->SetRightMargin(0.15);
	  phPtYTmp->Draw();
	  pgPtYEff_std->GetZaxis()->SetRangeUser(0.0,1.0);
	  pgPtYEff_std->Draw("colz same");
	  lx.DrawLatex(0.,2.,"Std_reco");
	  gPad->Update();
	  
	  pcPtYPairEff->cd(3);  
	  pcPtYPairEff->GetPad(3)->SetLeftMargin(0.16);
	  pcPtYPairEff->GetPad(3)->SetRightMargin(0.15);    
	  phPtYTmp->Draw();
	  pgPtYEff_regit->GetZaxis()->SetRangeUser(0.0,1.0);
	  pgPtYEff_regit->Draw("colz same");
	  lx.DrawLatex(0.,2,"Regit_reco");

	  if(bSavePlots)
	    {
	      TString outFileBase1(Form("%s/%s_%s",pOutFilePath,pcPtYPairEff->GetTitle(),pOutFileName));
	      TString outFileGif1 = outFileBase1+".gif";
	      pcPtYPairEff->Print(outFileGif1.Data(),"gifLandscape");
	      TString outFilePdf1 = outFileBase1+".pdf";
	      pcPtYPairEff->SaveAs(outFilePdf1.Data());
	      
	    }
	  
	}

      if(bSavePlots)
	{
	  TString outFileBase(Form("%s/%s_%s",pOutFilePath,pcPairEff->GetTitle(),pOutFileName));
	  TString outFileGif = outFileBase+".gif";
	  pcPairEff->Print(outFileGif.Data(),"gifLandscape");
	  TString outFilePdf = outFileBase+".pdf";
	  pcPairEff->SaveAs(outFilePdf.Data());
	  
	}
    }
  
  // ##########--------------------- single plots!!!
  if(bDoSinglePlots)
    {
      TChain *ptTrkPp   = new TChain(Form("%s",pTupleTrkStd));
      TChain *ptTrkStd   = new TChain(Form("%s",pTupleTrkStd));
      TChain *ptTrkRegit = new TChain(Form("%s",pTupleTrkRegit));
      //nPtBins
      for(int iptbin=1; iptbin<nPtBins; iptbin++)
	{
	  const char* infileTrkPp   = Form("%s%s.root",aInFileLocation[1],aPtSignalBins[iptbin]);
	  ptTrkPp->Add(infileTrkPp);
	  const char* infileTrkStd = Form("%s%s.root",aInFileLocation[2],aPtSignalBins[iptbin]);
	  ptTrkStd->Add(infileTrkStd);
	  const char* infileTrkRegit = Form("%s%s.root",aInFileLocation[3],aPtSignalBins[iptbin]);
	  ptTrkRegit->Add(infileTrkRegit);
	}
      
     
      TH1D *phPtGenRecoTrk_pp   = new TH1D("phPtGenRecoTrk_pp","phPtGenRecoTrk_pp",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phPtGenTrk_pp       = new TH1D("phPtGenTrk_pp","phPtGenTrk_pp",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phEtaGenRecoTrk_pp  = new TH1D("phEtaGenRecoTrk_pp","phEtaGenRecoTrk_pp",12,-2.4,2.4);
      TH1D *phEtaGenTrk_pp      = new TH1D("phEtaGenTrk_pp","phEtaGenTrk_pp",12,-2.4,2.4);
      phPtGenRecoTrk_pp->Sumw2();
      phPtGenTrk_pp->Sumw2();
      phEtaGenRecoTrk_pp->Sumw2();
      phEtaGenTrk_pp->Sumw2();
     
      TH1D *phPtGenRecoTrk_std   = new TH1D("phPtGenRecoTrk_std","phPtGenRecoTrk_std",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phPtGenTrk_std       = new TH1D("phPtGenTrk_std","phPtGenTrk_std",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phEtaGenRecoTrk_std  = new TH1D("phEtaGenRecoTrk_std","phEtaGenRecoTrk_std",12,-2.4,2.4);
      TH1D *phEtaGenTrk_std      = new TH1D("phEtaGenTrk_std","phEtaGenTrk_std",12,-2.4,2.4);
      phPtGenRecoTrk_std->Sumw2();
      phPtGenTrk_std->Sumw2();
      phEtaGenRecoTrk_std->Sumw2();
      phEtaGenTrk_std->Sumw2();

      TH1D *phPtGenRecoTrk_regit = new TH1D("phPtGenRecoTrk_regit","phPtGenRecoTrk_regit",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phPtGenTrk_regit     = new TH1D("phPtGenTrk_regit","phPtGenTrk_regit",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phEtaGenRecoTrk_regit= new TH1D("phEtaGenRecoTrk_regit","phEtaGenRecoTrk_regit",12,-2.4,2.4);
      TH1D *phEtaGenTrk_regit    = new TH1D("phEtaGenTrk_regit","phEtaGenTrk_regit",12,-2.4,2.4);
      phPtGenRecoTrk_regit->Sumw2();
      phPtGenTrk_regit->Sumw2();
      phEtaGenRecoTrk_regit->Sumw2();
      phEtaGenTrk_regit->Sumw2();
  
      TCanvas *pcTemp2 = new TCanvas("pcTemp2","pcTemp2", 900, 400);
      pcTemp2->Divide(2,1);
      pcTemp2->cd(1);
      pcTemp2->cd(1);
      ptTrkPp->Draw("pt>>phPtGenTrk_pp",trkGenSelection,"E");
      ptTrkPp->Draw("ptreco>>phPtGenRecoTrk_pp",trkRecoSelection,"Esame");
      pcTemp2->cd(2);
      ptTrkPp->Draw("eta>>phEtaGenTrk_pp",trkGenSelection,"E");
      ptTrkPp->Draw("etareco>>phEtaGenRecoTrk_pp",trkRecoSelection,"Esame");
      
      ptTrkStd->Draw("pt>>phPtGenTrk_std",trkGenSelection,"Esame");
      ptTrkStd->Draw("ptreco>>phPtGenRecoTrk_std",trkRecoSelection,"Esame");
      pcTemp2->cd(2);
      ptTrkStd->Draw("eta>>phEtaGenTrk_std",trkGenSelection,"Esame");
      ptTrkStd->Draw("etareco>>phEtaGenRecoTrk_std",trkRecoSelection,"Esame");
      
      pcTemp2->cd(1);
      ptTrkRegit->Draw("pt>>phPtGenTrk_regit",trkGenSelection,"Esame");
      ptTrkRegit->Draw("ptreco>>phPtGenRecoTrk_regit",trkRecoSelection,"Esame");
      pcTemp2->cd(2);
      ptTrkRegit->Draw("eta>>phEtaGenTrk_regit",trkGenSelection,"Esame");
      ptTrkRegit->Draw("etareco>>phEtaGenRecoTrk_regit",trkRecoSelection,"Esame");

      TH1D *pgPtTrkEff_pp = new TH1D("pgPtTrkEff_pp","pgPtTrkEff_pp",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *pgEtaTrkEff_pp  = new TH1D("pgEtaTrkEff_pp","pgEtaTrkEff_pp",12,-2.4,2.4);
      pgPtTrkEff_pp->Sumw2();
      pgEtaTrkEff_pp->Sumw2();
      TH1D *pgPtTrkEff_std = new TH1D("pgPtTrkEff_std","pgPtTrkEff_std",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *pgEtaTrkEff_std  = new TH1D("pgEtaTrkEff_std","pgEtaTrkEff_std",12,-2.4,2.4);
      pgPtTrkEff_std->Sumw2();
      pgEtaTrkEff_std->Sumw2();
      TH1D *pgPtTrkEff_regit =  new TH1D("pgPtTrkEff_regit","pgPtTrkEff_regit",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *pgEtaTrkEff_regit  =  new TH1D("pgEtaTrkEff_regit","pgEtaTrkEff_regit",12,-2.4,2.4);
      pgPtTrkEff_regit->Sumw2();
      pgEtaTrkEff_regit->Sumw2();

      pgPtTrkEff_pp->Divide(phPtGenRecoTrk_pp,phPtGenTrk_pp,1,1,"b");
      pgEtaTrkEff_pp->Divide(phEtaGenRecoTrk_pp,phEtaGenTrk_pp,1,1,"b");
      pgPtTrkEff_std->Divide(phPtGenRecoTrk_std,phPtGenTrk_std,1,1,"b");
      pgEtaTrkEff_std->Divide(phEtaGenRecoTrk_std,phEtaGenTrk_std,1,1,"b");
      pgPtTrkEff_regit->Divide(phPtGenRecoTrk_regit,phPtGenTrk_regit,1,1,"b");
      pgEtaTrkEff_regit->Divide(phEtaGenRecoTrk_regit,phEtaGenTrk_regit,1,1,"b");
  
      // 2D stuff
      TH2F *phPtEtaGenRecoTrk_pp  = new TH2F("phPtEtaGenRecoTrk_pp",";#eta;p_{T}^{#mu}[GeV/c]",48,-2.4,2.4,nPtTrkBins,ptTrkMin,ptTrkMax);
      TH2F *phPtEtaGenTrk_pp      = new TH2F("phPtEtaGenTrk_pp",";#eta;p_{T}^{#mu}[GeV/c]",48,-2.4,2.4,nPtTrkBins,ptTrkMin,ptTrkMax);
      ptTrkPp->Draw("pt:eta>>phPtEtaGenTrk_pp",trkGenSelection,"colz");
      ptTrkPp->Draw("ptreco:etareco>>phPtEtaGenRecoTrk_pp",trkRecoSelection,"colz");
      TH2F *phPtEtaGenRecoTrk_std  = new TH2F("phPtEtaGenRecoTrk_std",";#eta;p_{T}^{#mu}[GeV/c]",48,-2.4,2.4,nPtTrkBins,ptTrkMin,ptTrkMax);
      TH2F *phPtEtaGenTrk_std      = new TH2F("phPtEtaGenTrk_std",";#eta;p_{T}^{#mu}[GeV/c]",48,-2.4,2.4,nPtTrkBins,ptTrkMin,ptTrkMax);
      ptTrkStd->Draw("pt:eta>>phPtEtaGenTrk_std",trkGenSelection,"colz");
      ptTrkStd->Draw("ptreco:etareco>>phPtEtaGenRecoTrk_std",trkRecoSelection,"colz");
      TH2F *phPtEtaGenRecoTrk_regit= new TH2F("phPtEtaGenRecoTrk_regit",";#eta;p_{T}^{#mu#mu}[GeV/c]",48,-2.4,2.4,nPtTrkBins,ptTrkMin,ptTrkMax);
      TH2F *phPtEtaGenTrk_regit    = new TH2F("phPtEtaGenTrk_regit",";#eta;p_{T}^{#mu#mu}[GeV/c]",48,-2.4,2.4,nPtTrkBins,ptTrkMin,ptTrkMax);
      ptTrkRegit->Draw("pt:eta>>phPtEtaGenTrk_regit",trkGenSelection,"colz");
      ptTrkRegit->Draw("ptreco:etareco>>phPtEtaGenRecoTrk_regit",trkRecoSelection,"colz");

      TH2D *pgPtEtaEff_pp   = new TH2D("pgPtEtaEff_pp",";#eta;p_{T}^{#mu}[GeV/c]",48,-2.4,2.4,nPtTrkBins,ptTrkMin,ptTrkMax);
      TH2D *pgPtEtaEff_std   = new TH2D("pgPtEtaEff_std",";#eta;p_{T}^{#mu}[GeV/c]",48,-2.4,2.4,nPtTrkBins,ptTrkMin,ptTrkMax);
      TH2D *pgPtEtaEff_regit = new TH2D("pgPtEtaEff_regit",";#eta;p_{T}^{#mu}[GeV/c]",48,-2.4,2.4,nPtTrkBins,ptTrkMin,ptTrkMax);
      pgPtEtaEff_pp->Divide(phPtEtaGenRecoTrk_pp,phPtEtaGenTrk_pp,1,1,"b");
      pgPtEtaEff_std->Divide(phPtEtaGenRecoTrk_std,phPtEtaGenTrk_std,1,1,"b");
      pgPtEtaEff_regit->Divide(phPtEtaGenRecoTrk_regit,phPtEtaGenTrk_regit,1,1,"b");


      // drawing
      pgPtTrkEff_pp->SetLineColor(3);
      pgPtTrkEff_pp->SetMarkerColor(3);
      pgPtTrkEff_pp->SetMarkerStyle(3);
      
      pgEtaTrkEff_pp->SetLineColor(3);
      pgEtaTrkEff_pp->SetMarkerColor(3);
      pgEtaTrkEff_pp->SetMarkerStyle(3);
      
      pgPtTrkEff_std->SetLineColor(4);
      pgPtTrkEff_std->SetMarkerColor(4);
      pgPtTrkEff_std->SetMarkerStyle(4);
      
      pgEtaTrkEff_std->SetLineColor(4);
      pgEtaTrkEff_std->SetMarkerColor(4);
      pgEtaTrkEff_std->SetMarkerStyle(4);
      
      pgPtTrkEff_regit->SetLineColor(2);
      pgPtTrkEff_regit->SetMarkerColor(2);
      pgPtTrkEff_regit->SetMarkerStyle(25);
      
      pgEtaTrkEff_regit->SetLineColor(2);
      pgEtaTrkEff_regit->SetMarkerColor(2);
      pgEtaTrkEff_regit->SetMarkerStyle(25);
      
      TCanvas *pcTrkEff = new TCanvas("pcTrkEff","pcTrkEff",1000,600);
      pcTrkEff->Divide(2,1);
      pcTrkEff->cd(1); //gPad->SetLogy();
      phPtTmp2->Draw();
      pgPtTrkEff_pp->Draw("pl same");
      pgPtTrkEff_std->Draw("pl same");
      pgPtTrkEff_regit->Draw("pl same");
      
      pcTrkEff->cd(2); //gPad->SetLogy();
      phEtaTmp->Draw();
      pgEtaTrkEff_pp->Draw("pl same");
      pgEtaTrkEff_std->Draw("pl same");
      pgEtaTrkEff_regit->Draw("pl same");
      
      TLegend *t2 = new TLegend(0.34,0.78,0.94,0.91);
      t2->SetBorderSize(0);
      t2->SetFillStyle(0);
      t2->AddEntry(pgPtTrkEff_pp,"Pp_reco","pl");
      t2->AddEntry(pgPtTrkEff_std,"Std_reco","pl");
      t2->AddEntry(pgPtTrkEff_regit,"Regit_reco","pl");
      t2->Draw();
      
      // 2d plots
      //2D histo
      if(bDo2D)
	{
	  TLatex lx2;
	  TCanvas *pcPtEtaTrkEff = new TCanvas("pcPtEtaTrkEff","pcPtEtaTrkEff",1300,500);
	  pcPtEtaTrkEff->Divide(3,1);
	  pcPtEtaTrkEff->cd(1);      
	  pcPtEtaTrkEff->GetPad(1)->SetLeftMargin(0.16);
	  pcPtEtaTrkEff->GetPad(1)->SetRightMargin(0.15); 
	  phPtEtaTmp->Draw();
	  pgPtEtaEff_pp->GetZaxis()->SetRangeUser(0.0,1.0);
	  pgPtEtaEff_pp->Draw("COLZ same");
	  lx2.DrawLatex(-0.2,2,"Pp_reco");
	  
	  
	  pcPtEtaTrkEff->cd(2);      
	  pcPtEtaTrkEff->GetPad(2)->SetLeftMargin(0.16);
	  pcPtEtaTrkEff->GetPad(2)->SetRightMargin(0.15); 
	  phPtEtaTmp->Draw();
	  pgPtEtaEff_std->GetZaxis()->SetRangeUser(0.0,1.0);
	  pgPtEtaEff_std->Draw("COLZ same");
	  lx2.DrawLatex(-0.2,2,"Std_reco");
	  
	  pcPtEtaTrkEff->cd(3);
	  pcPtEtaTrkEff->GetPad(3)->SetLeftMargin(0.16);
	  pcPtEtaTrkEff->GetPad(3)->SetRightMargin(0.15);       
	  phPtEtaTmp->Draw();
	  pgPtEtaEff_regit->GetZaxis()->SetRangeUser(0.0,1.0);
	  pgPtEtaEff_regit->Draw("colz same");
	  lx2.DrawLatex(-0.2,2,"Regit_reco");

     if(bSavePlots)
     {
        TString outFileBase2(Form("%s/%s_%s",pOutFilePath,pcPtEtaTrkEff->GetTitle(),pOutFileName));
        TString outFileGif2 = outFileBase2+".gif";
        pcPtEtaTrkEff->Print(outFileGif2.Data(),"gifLandscape");
        TString outFilePdf2 = outFileBase2+".pdf";
        pcPtEtaTrkEff->SaveAs(outFilePdf2.Data());
     }

   }
      if(bSavePlots)
      {
         TString outFileBase(Form("%s/%s_%s",pOutFilePath,pcTrkEff->GetTitle(),pOutFileName));
         TString outFileGif = outFileBase+".gif";
         pcTrkEff->Print(outFileGif.Data(),"gifLandscape");
         TString outFilePdf2 = outFileBase+".pdf";
         pcTrkEff->SaveAs(outFilePdf2.Data());
      }
    }


}


//____________________________________________________________________________________
