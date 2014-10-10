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
static bool smuacc_glb(double pt, double eta)
{
   if (fabs(eta)<1 && pt>3.4) return true;
   else if (fabs(eta)>=1 && fabs(eta)<1.5 && pt>5.8-2.4*fabs(eta)) return true;
   else if (fabs(eta)>=1.5 && fabs(eta)<2.4 && pt>3.6667-(7./9.)*fabs(eta)) return true;
   else return false;
}
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
//____________________________________________________________________________________
void makeRecoRatios(const char* pTuplePairPp="mcmatchanalysis/pnSimRecoPairType1",//1
		    const char* pTuplePairStd="mcmatchanalysis/pnSimRecoPairType2",//2
		    const char* pTuplePairRegit="mcmatchanalysis/pnSimRecoPairType1",//2
		    const char* pTupleTrkPp="mcmatchanalysis/pnSimRecoTrkType1",
		    const char* pTupleTrkStd="mcmatchanalysis/pnSimRecoTrkType2",
		    const char* pTupleTrkRegit="mcmatchanalysis/pnSimRecoTrkType1",
		    const char* pOutFilePath="outputfigs",
		    const char* pOutFileName="jpsi_myregit",
		    Bool_t bSavePlots=true)
{
  // gROOT->Macro("/afs/cern.ch/user/m/mironov/utilities/setStyle.C+");
  gStyle->SetPalette(1);
  const char* aInFileLocation[4] = {"",
				    "/afs/cern.ch/user/e/echapon/workspace/private/jpsi_PbPb_5_3_17/CMSSW_5_3_17/src/RegitStudy/McAnalyzer/test/ntuples_jpsi/ntuples_regit_embd",
				    "/afs/cern.ch/user/e/echapon/workspace/private/jpsi_PbPb_5_3_17/CMSSW_5_3_17/src/RegitStudy/McAnalyzer/test/ntuples_jpsi/ntuples_regit_embd",
                "/afs/cern.ch/user/e/echapon/workspace/private/jpsi_PbPb_5_3_17/CMSSW_5_3_17/src/RegitStudy/McAnalyzer/test/ntuples_jpsi/ntuples_myregit_embd"
				   };
  const char* aRecoNames[3] = {"std HI", "RegIt (5.3.17)", "RegIt (new)"};
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
  bool bDoEta         =  true;
  bool bDoPairPlots   =  true;
  float ptPairMin = 0;
  float ptPairMax = 10;
  int nPtPairBins = 10;

  float ptTrkMin = 0;
  float ptTrkMax = 10;
  int nPtTrkBins = 10;

  float lxyPairMin = 0;
  float lxyPairMax = 1;
  int nLxyPairBins = 10;

  float lxyTrkMin = 0;
  float lxyTrkMax = 1;
  int nLxyTrkBins = 10;

  // modified
  TCut pairGenSelection("smuacc(pt1,eta1)&&smuacc(pt2,eta2) &&npixelhits1>1&&npixelhits2>1 &&nmuonhits1>0&&nmuonhits2>0");
  TCut pairRecoSelection("smuacc(ptreco1,etareco1)&&smuacc(ptreco2,etareco2) &&nvalidpixelhitsreco1>1&&nvalidpixelhitsreco2>1 &&nvalidmuonhitsreco1>0&&nvalidmuonhitsreco2>0 &&minvreco>2.6&&minvreco<3.5&&chi2ndofreco1<4&&chi2ndofreco2<4");

  TCut trkGenSelection("smuacc(pt,eta)&&(abs(idparent)<550&&abs(idparent)>442) &&npixelhits>1&&nmuonhits>0");
  // TCut trkRecoSelection("smuacc(ptreco,etareco)&&abs(etareco)<2.4&&(abs(idparent)<550&&abs(idparent)>442) &&nvalidpixelhitsreco>1&&nvalidmuonhitsreco>0 &&nmatch>0&&chi2ndofreco<4");
  TCut trkRecoSelection("smuacc(ptreco,etareco)&&(abs(idparent)<550&&abs(idparent)>442) &&nvalidpixelhitsreco>1&&nvalidmuonhitsreco>0 &&nmatch>0");

  TCut trkGenSelection_barrel("smuacc(pt,eta)&&abs(eta)<1.2&&(abs(idparent)<550&&abs(idparent)>442) &&npixelhits>1&&nmuonhits>0");
  TCut trkRecoSelection_barrel("smuacc(ptreco,etareco)&&abs(etareco)<1.2&&(abs(idparent)<550&&abs(idparent)>442) &&nvalidpixelhitsreco>1&&nvalidmuonhitsreco>0 &&nmatch>0&&chi2ndofreco<4");
  TCut trkGenSelection_inter("smuacc(pt,eta)&&abs(eta)>1.2&&abs(eta)<1.6(abs(idparent)<550&&abs(idparent)>442) &&npixelhits>1&&nmuonhits>0");
  TCut trkRecoSelection_inter("smuacc(ptreco,etareco)&&abs(etareco)>1.2&&abs(etareco)<1.6&&(abs(idparent)<550&&abs(idparent)>442) &&nvalidpixelhitsreco>1&&nvalidmuonhitsreco>0 &&nmatch>0&&chi2ndofreco<4");
  TCut trkGenSelection_fwd("smuacc(pt,eta)&&abs(eta)>1.6&&abs(eta)<2.4&&(abs(idparent)<550&&abs(idparent)>442) &&npixelhits>1&&nmuonhits>0");
  TCut trkRecoSelection_fwd("smuacc(ptreco,etareco)&&abs(etareco)>1.6&&abs(etareco)<2.4&&(abs(idparent)<550&&abs(idparent)>442) &&nvalidpixelhitsreco>1&&nvalidmuonhitsreco>0 &&nmatch>0&&chi2ndofreco<4");

  // TCut pairGenSelection("TMath::Abs(eta1)<2.4&&TMath::Abs(eta2)<2.4&&npixelhits1>1&&npixelhits2>1&&nmuonhits1>0&&nmuonhits2>0");
//   TCut pairRecoSelection("nmuonhits1>0&&nmuonhits2>0&&minvreco>2.6&&minvreco<3.5");
//   TCut trkGenSelection("TMath::Abs(eta)<2.4&&idparent==443&&npixelhits>1&&nmuonhits>0");
//   TCut trkRecoSelection("nvalidmuonhitsreco>0");
  
  // axis pair
  TH1F *phPtTmp = new TH1F("phPtTmp",";p_{T}^{#mu#mu}[GeV/c];Efficiency",1,ptPairMin,ptPairMax);
  TH1F *phLxyTmp = new TH1F("phLxyTmp",";IP;Efficiency",1,lxyPairMin,lxyPairMax);
  TH1F *phYTmp  = new TH1F("phYTmp",";y^{#mu#mu};Efficiency",1,-2.4,2.4);
  phPtTmp->SetMinimum(0.5);
  phLxyTmp->SetMinimum(0.5);
  phYTmp->SetMinimum(0.5);
  phPtTmp->SetMaximum(1.);
  phLxyTmp->SetMaximum(1.);
  phYTmp->SetMaximum(1.);
  TH2F *phPtYTmp = new TH2F("phPtYTmp",";y;p_{T}^{#mu#mu}[GeV/c]",1,-2.4,2.4,1,ptPairMin,ptPairMax);
 

  // axis single
  TH1F *phPtTmp2 = new TH1F("phPtTmp2",";p_{T}^{#mu}[GeV/c];Efficiency",1,ptTrkMin,ptTrkMax);
  phPtTmp2->SetMinimum(0.5);
  phPtTmp2->SetMaximum(1.);
    
  TH1F *phLxyTmp2 = new TH1F("phLxyTmp2",";IP;Efficiency",1,lxyTrkMin,lxyTrkMax);
  phLxyTmp2->SetMinimum(0.5);
  phLxyTmp2->SetMaximum(1.);
    
  TH1F *phEtaTmp  = new TH1F("phEtaTmp",";#eta^{#mu};Efficiency",1,-2.4,2.4);
  phEtaTmp->SetMinimum(0.5);
  phEtaTmp->SetMaximum(1.);
  TH2F *phPtEtaTmp = new TH2F("phPtEtaTmp",";#eta;p_{T}^{#mu}[GeV/c]",1,-2.4,2.4,1,ptTrkMin,ptTrkMax);
  // TH2F *phPtEtaTmp = new TH2F("phPtEtaTmp",";#eta;p_{T}^{#mu}[GeV/c]",1,-2.4,2.4,1,ptTrkMin,5);

  if(bDoPairPlots)
    {
      TChain *ptPairPp   = new TChain(Form("%s",pTuplePairPp));
      TChain *ptPairStd   = new TChain(Form("%s",pTuplePairStd));
      TChain *ptPairRegit = new TChain(Form("%s",pTuplePairRegit));
      for(int iptbin=1; iptbin<nPtBins; iptbin++)
	{
	  const char* infilePairPp   = Form("%s%s.root",aInFileLocation[1],aPtSignalBins[iptbin]);
	  ptPairPp->Add(infilePairPp);
	  const char* infilePairStd   = Form("%s%s.root",aInFileLocation[2],aPtSignalBins[iptbin]);
	  ptPairStd->Add(infilePairStd);
	  const char* infilePairRegit = Form("%s%s.root",aInFileLocation[3],aPtSignalBins[iptbin]);
	  ptPairRegit->Add(infilePairRegit);
	}
      ptPairPp->SetAlias("lxy","sqrt((vtxz1-bx)*(vtxz1-bx)+(vtxy1-by)*(vtxy1-by))");
      ptPairStd->SetAlias("lxy","sqrt((vtxz1-bx)*(vtxz1-bx)+(vtxy1-by)*(vtxy1-by))");
      ptPairRegit->SetAlias("lxy","sqrt((vtxz1-bx)*(vtxz1-bx)+(vtxy1-by)*(vtxy1-by))");
   
  
      // ########## pair plots!!!
      TH1D *phPtGenRecoPair_pp = new TH1D("phPtGenRecoPair_pp","phPtGenRecoPair_pp",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *phLxyGenRecoPair_pp = new TH1D("phLxyGenRecoPair_pp","phLxyGenRecoPair_pp",nLxyPairBins,lxyPairMin,lxyPairMax);
      TH1D *phYGenRecoPair_pp  = new TH1D("phYGenRecoPair_pp","phYGenRecoPair_pp",12,-2.4,2.4);
      phPtGenRecoPair_pp->Sumw2();
      phYGenRecoPair_pp->Sumw2();
            
      TH1D *phPtGenRecoPair_std = new TH1D("phPtGenRecoPair_std","phPtGenRecoPair_std",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *phLxyGenRecoPair_std = new TH1D("phLxyGenRecoPair_std","phLxyGenRecoPair_std",nLxyPairBins,lxyPairMin,lxyPairMax);
      TH1D *phYGenRecoPair_std  = new TH1D("phYGenRecoPair_std","phYGenRecoPair_std",12,-2.4,2.4);
      phPtGenRecoPair_std->Sumw2();
      phYGenRecoPair_std->Sumw2();
            
      TH1D *phPtGenRecoPair_regit = new TH1D("phPtGenRecoPair_regit","phPtGenRecoPair_regit",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *phLxyGenRecoPair_regit = new TH1D("phLxyGenRecoPair_regit","phLxyGenRecoPair_regit",nLxyPairBins,lxyPairMin,lxyPairMax);
      TH1D *phYGenRecoPair_regit  = new TH1D("phYGenRecoPair_regit","phYGenRecoPair_regit",12,-2.4,2.4);
      phPtGenRecoPair_regit->Sumw2();
      phYGenRecoPair_regit->Sumw2();
            
            
      TCanvas *pcTemp = new TCanvas("pcTemp","pcTemp", 1300, 400);
      pcTemp->Divide(3,1);
      pcTemp->cd(1);
      ptPairStd->Draw("pt>>phPtGenRecoPair_std",pairRecoSelection,"Esame");
      ptPairPp->Draw("pt>>phPtGenRecoPair_pp",pairRecoSelection,"Esame");
      ptPairRegit->Draw("pt>>phPtGenRecoPair_regit",pairRecoSelection,"Esame");
      pcTemp->cd(2);
      ptPairStd->Draw("y>>phYGenRecoPair_std",pairRecoSelection,"Esame");
      ptPairPp->Draw("y>>phYGenRecoPair_pp",pairRecoSelection,"Esame");
      ptPairRegit->Draw("y>>phYGenRecoPair_regit",pairRecoSelection,"Esame");
      pcTemp->cd(3);
      ptPairStd->Draw("lxy>>phLxyGenRecoPair_std",pairRecoSelection,"Esame");
      ptPairPp->Draw("lxy>>phLxyGenRecoPair_pp",pairRecoSelection,"Esame");
      ptPairRegit->Draw("lxy>>phLxyGenRecoPair_regit",pairRecoSelection,"Esame");
      
      TH1D *pgPtEff = new TH1D("pgPtEff","pgPtEff",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *pgLxyEff = new TH1D("pgLxyEff","pgLxyEff",nLxyPairBins,lxyPairMin,lxyPairMax);
      TH1D *pgYEff  = new TH1D("pgYEff","pgYEff",12,-2.4,2.4);
      TH1D *pgPtEff2 = new TH1D("pgPtEff2","pgPtEff2",nPtPairBins,ptPairMin,ptPairMax);
      TH1D *pgLxyEff2 = new TH1D("pgLxyEff2","pgLxyEff2",nLxyPairBins,lxyPairMin,lxyPairMax);
      TH1D *pgYEff2  = new TH1D("pgYEff2","pgYEff2",12,-2.4,2.4);
      pgPtEff->Sumw2();
      pgLxyEff->Sumw2();
      pgYEff->Sumw2();
      pgPtEff2->Sumw2();
      pgLxyEff2->Sumw2();
      pgYEff2->Sumw2();
    
      pgPtEff->Divide(phPtGenRecoPair_std,phPtGenRecoPair_regit,1,1,"b");
      pgLxyEff->Divide(phLxyGenRecoPair_std,phLxyGenRecoPair_regit,1,1,"b");
      pgYEff->Divide(phYGenRecoPair_std,phYGenRecoPair_regit,1,1,"b");
      pgPtEff2->Divide(phPtGenRecoPair_regit,phPtGenRecoPair_pp,1,1,"b");
      pgLxyEff2->Divide(phLxyGenRecoPair_regit,phLxyGenRecoPair_pp,1,1,"b");
      pgYEff2->Divide(phYGenRecoPair_regit,phYGenRecoPair_pp,1,1,"b");
  
      // drawing
      pgPtEff->SetLineColor(4);
      pgPtEff->SetMarkerColor(4);
      pgPtEff->SetMarkerStyle(4);
      pgPtEff2->SetLineColor(3);
      pgPtEff2->SetMarkerColor(3);
      pgPtEff2->SetMarkerStyle(3);
      
      pgLxyEff->SetLineColor(4);
      pgLxyEff->SetMarkerColor(4);
      pgLxyEff->SetMarkerStyle(4);
      pgLxyEff2->SetLineColor(3);
      pgLxyEff2->SetMarkerColor(3);
      pgLxyEff2->SetMarkerStyle(3);
      
      pgYEff->SetLineColor(4);
      pgYEff->SetMarkerColor(4);
      pgYEff->SetMarkerStyle(4);
      pgYEff2->SetLineColor(3);
      pgYEff2->SetMarkerColor(3);
      pgYEff2->SetMarkerStyle(3);
      
      TCanvas *pcPairEff = new TCanvas("pcPairEff","pcPairEff",900,500);
      pcPairEff->Divide(3,1);
      pcPairEff->cd(1);// gPad->SetLogy();
      phPtTmp->Draw();
      pgPtEff->Draw("pl same");
      pgPtEff2->Draw("pl same");
      
      pcPairEff->cd(2);// gPad->SetLogy();
      phLxyTmp->Draw();
      pgLxyEff->Draw("pl same");
      pgLxyEff2->Draw("pl same");
      
      pcPairEff->cd(3); //gPad->SetLogy();
      phYTmp->Draw();
      pgYEff->Draw("pl same");
      pgYEff2->Draw("pl same");
      
      TLegend *t = new TLegend(0.34,0.78,0.94,0.91);
      t->SetBorderSize(0);
      t->SetFillStyle(0);
      // t->AddEntry(pgPtEff,"Std_reco/Regit_reco","pl");
      t->AddEntry(pgPtEff,Form("%s / %s",aRecoNames[1],aRecoNames[2]),"pl");
      // t->AddEntry(pgPtEff2,"Regit_reco/Pp_reco","pl");
      t->AddEntry(pgPtEff2,Form("%s / %s",aRecoNames[2],aRecoNames[0]),"pl");
      t->Draw();

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
      TChain *ptTrkPp   = new TChain(Form("%s",pTupleTrkPp));
      TChain *ptTrkStd   = new TChain(Form("%s",pTupleTrkStd));
      TChain *ptTrkRegit = new TChain(Form("%s",pTupleTrkRegit));
      //nPtBins
      for(int iptbin=1; iptbin<nPtBins; iptbin++)
	{
	  const char* infileTrkPp   = Form("%s%s.root",aInFileLocation[1],aPtSignalBins[iptbin]);
	  ptTrkPp->Add(infileTrkPp);
	  const char* infileTrkStd   = Form("%s%s.root",aInFileLocation[2],aPtSignalBins[iptbin]);
	  ptTrkStd->Add(infileTrkStd);
	  const char* infileTrkRegit = Form("%s%s.root",aInFileLocation[3],aPtSignalBins[iptbin]);
	  ptTrkRegit->Add(infileTrkRegit);
	}
      ptTrkPp->SetAlias("lxy","sqrt((vtxz-bx)*(vtxz-bx)+(vtxy-by)*(vtxy-by))");
      ptTrkStd->SetAlias("lxy","sqrt((vtxz-bx)*(vtxz-bx)+(vtxy-by)*(vtxy-by))");
      ptTrkRegit->SetAlias("lxy","sqrt((vtxz-bx)*(vtxz-bx)+(vtxy-by)*(vtxy-by))");
      
      // full coverage     
      TH1D *phPtGenRecoTrk_pp   = new TH1D("phPtGenRecoTrk_pp","phPtGenRecoTrk_pp",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phPtGenTrk_pp       = new TH1D("phPtGenTrk_pp","phPtGenTrk_pp",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phLxyGenRecoTrk_pp   = new TH1D("phLxyGenRecoTrk_pp","phLxyGenRecoTrk_pp",nLxyTrkBins,lxyTrkMin,lxyTrkMax);
      TH1D *phLxyGenTrk_pp       = new TH1D("phLxyGenTrk_pp","phLxyGenTrk_pp",nLxyTrkBins,lxyTrkMin,lxyTrkMax);
      TH1D *phEtaGenRecoTrk_pp  = new TH1D("phEtaGenRecoTrk_pp","phEtaGenRecoTrk_pp",12,-2.4,2.4);
      TH1D *phEtaGenTrk_pp      = new TH1D("phEtaGenTrk_pp","phEtaGenTrk_pp",12,-2.4,2.4);
      phPtGenRecoTrk_pp->Sumw2();
      phPtGenTrk_pp->Sumw2();
      phLxyGenRecoTrk_pp->Sumw2();
      phLxyGenTrk_pp->Sumw2();
      phEtaGenRecoTrk_pp->Sumw2();
      phEtaGenTrk_pp->Sumw2();

      TH1D *phPtGenRecoTrk_std   = new TH1D("phPtGenRecoTrk_std","phPtGenRecoTrk_std",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phPtGenTrk_std       = new TH1D("phPtGenTrk_std","phPtGenTrk_std",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phLxyGenRecoTrk_std   = new TH1D("phLxyGenRecoTrk_std","phLxyGenRecoTrk_std",nLxyTrkBins,lxyTrkMin,lxyTrkMax);
      TH1D *phLxyGenTrk_std       = new TH1D("phLxyGenTrk_std","phLxyGenTrk_std",nLxyTrkBins,lxyTrkMin,lxyTrkMax);
      TH1D *phEtaGenRecoTrk_std  = new TH1D("phEtaGenRecoTrk_std","phEtaGenRecoTrk_std",12,-2.4,2.4);
      TH1D *phEtaGenTrk_std      = new TH1D("phEtaGenTrk_std","phEtaGenTrk_std",12,-2.4,2.4);
      phPtGenRecoTrk_std->Sumw2();
      phPtGenTrk_std->Sumw2();
      phLxyGenRecoTrk_std->Sumw2();
      phLxyGenTrk_std->Sumw2();
      phEtaGenRecoTrk_std->Sumw2();
      phEtaGenTrk_std->Sumw2();

      TH1D *phPtGenRecoTrk_regit = new TH1D("phPtGenRecoTrk_regit","phPtGenRecoTrk_regit",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phPtGenTrk_regit     = new TH1D("phPtGenTrk_regit","phPtGenTrk_regit",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *phLxyGenRecoTrk_regit = new TH1D("phLxyGenRecoTrk_regit","phLxyGenRecoTrk_regit",nLxyTrkBins,lxyTrkMin,lxyTrkMax);
      TH1D *phLxyGenTrk_regit       = new TH1D("phLxyGenTrk_regit","phLxyGenTrk_regit",nLxyTrkBins,lxyTrkMin,lxyTrkMax);
      TH1D *phEtaGenRecoTrk_regit= new TH1D("phEtaGenRecoTrk_regit","phEtaGenRecoTrk_regit",12,-2.4,2.4);
      TH1D *phEtaGenTrk_regit    = new TH1D("phEtaGenTrk_regit","phEtaGenTrk_regit",12,-2.4,2.4);
      phPtGenRecoTrk_regit->Sumw2();
      phPtGenTrk_regit->Sumw2();
      phLxyGenRecoTrk_regit->Sumw2();
      phLxyGenTrk_regit->Sumw2();
      phEtaGenRecoTrk_regit->Sumw2();
      phEtaGenTrk_regit->Sumw2();

      //----------------------------------------------------------------------------------------------------
      TCanvas *pcTemp2 = new TCanvas("pcTemp2","pcTemp2");
      ptTrkPp->Draw("pt>>phPtGenTrk_pp",trkGenSelection,"E");
      ptTrkPp->Draw("ptreco>>phPtGenRecoTrk_pp",trkRecoSelection,"Esame");
      ptTrkPp->Draw("lxy>>phLxyGenTrk_pp",trkGenSelection,"Esame");
      ptTrkPp->Draw("lxy>>phLxyGenRecoTrk_pp",trkRecoSelection,"Esame");
      ptTrkPp->Draw("eta>>phEtaGenTrk_pp",trkGenSelection,"Esame");
      ptTrkPp->Draw("etareco>>phEtaGenRecoTrk_pp",trkRecoSelection,"Esame");
           
      ptTrkStd->Draw("pt>>phPtGenTrk_std",trkGenSelection,"Esame");
      ptTrkStd->Draw("ptreco>>phPtGenRecoTrk_std",trkRecoSelection,"Esame");
      ptTrkStd->Draw("lxy>>phLxyGenTrk_std",trkGenSelection,"Esame");
      ptTrkStd->Draw("lxy>>phLxyGenRecoTrk_std",trkRecoSelection,"Esame");
      ptTrkStd->Draw("eta>>phEtaGenTrk_std",trkGenSelection,"Esame");
      ptTrkStd->Draw("etareco>>phEtaGenRecoTrk_std",trkRecoSelection,"Esame");
           
      ptTrkRegit->Draw("pt>>phPtGenTrk_regit",trkGenSelection,"Esame");
      ptTrkRegit->Draw("ptreco>>phPtGenRecoTrk_regit",trkRecoSelection,"Esame");
      ptTrkRegit->Draw("lxy>>phLxyGenTrk_regit",trkGenSelection,"Esame");
      ptTrkRegit->Draw("lxy>>phLxyGenRecoTrk_regit",trkRecoSelection,"Esame");
      ptTrkRegit->Draw("eta>>phEtaGenTrk_regit",trkGenSelection,"Esame");
      ptTrkRegit->Draw("etareco>>phEtaGenRecoTrk_regit",trkRecoSelection,"Esame");
     
      // full coverage
      TH1D *pgPtTrkEff_pp = new TH1D("pgPtTrkEff_pp","pgPtTrkEff_pp",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *pgLxyTrkEff_pp = new TH1D("pgLxyTrkEff_pp","pgLxyTrkEff_pp",nLxyTrkBins,lxyTrkMin,lxyTrkMax);
      TH1D *pgEtaTrkEff_pp  = new TH1D("pgEtaTrkEff_pp","pgEtaTrkEff_pp",12,-2.4,2.4);
      pgPtTrkEff_pp->Sumw2();
      pgLxyTrkEff_pp->Sumw2();
      pgEtaTrkEff_pp->Sumw2();
      TH1D *pgPtTrkEff_std = new TH1D("pgPtTrkEff_std","pgPtTrkEff_std",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *pgLxyTrkEff_std = new TH1D("pgLxyTrkEff_std","pgLxyTrkEff_std",nLxyTrkBins,lxyTrkMin,lxyTrkMax);
      TH1D *pgEtaTrkEff_std  = new TH1D("pgEtaTrkEff_std","pgEtaTrkEff_std",12,-2.4,2.4);
      pgPtTrkEff_std->Sumw2();
      pgLxyTrkEff_std->Sumw2();
      pgEtaTrkEff_std->Sumw2();
      TH1D *pgPtTrkEff_regit =  new TH1D("pgPtTrkEff_regit","pgPtTrkEff_regit",nPtTrkBins,ptTrkMin,ptTrkMax);
      TH1D *pgLxyTrkEff_regit =  new TH1D("pgLxyTrkEff_regit","pgLxyTrkEff_regit",nLxyTrkBins,lxyTrkMin,lxyTrkMax);
      TH1D *pgEtaTrkEff_regit  =  new TH1D("pgEtaTrkEff_regit","pgEtaTrkEff_regit",12,-2.4,2.4);
      pgPtTrkEff_regit->Sumw2();
      pgLxyTrkEff_regit->Sumw2();
      pgEtaTrkEff_regit->Sumw2();

      pgPtTrkEff_pp->Divide(phPtGenRecoTrk_pp,phPtGenTrk_pp,1,1,"b");
      pgLxyTrkEff_pp->Divide(phLxyGenRecoTrk_pp,phLxyGenTrk_pp,1,1,"b");
      pgEtaTrkEff_pp->Divide(phEtaGenRecoTrk_pp,phEtaGenTrk_pp,1,1,"b");
      pgPtTrkEff_std->Divide(phPtGenRecoTrk_std,phPtGenTrk_std,1,1,"b");
      pgLxyTrkEff_std->Divide(phLxyGenRecoTrk_std,phLxyGenTrk_std,1,1,"b");
      pgEtaTrkEff_std->Divide(phEtaGenRecoTrk_std,phEtaGenTrk_std,1,1,"b");
      pgPtTrkEff_regit->Divide(phPtGenRecoTrk_regit,phPtGenTrk_regit,1,1,"b");
      pgLxyTrkEff_regit->Divide(phLxyGenRecoTrk_regit,phLxyGenTrk_regit,1,1,"b");
      pgEtaTrkEff_regit->Divide(phEtaGenRecoTrk_regit,phEtaGenTrk_regit,1,1,"b");
        
      // drawing
      pgPtTrkEff_pp->SetLineColor(3);
      pgPtTrkEff_pp->SetMarkerColor(3);
      pgPtTrkEff_pp->SetMarkerStyle(3);
      
      pgLxyTrkEff_pp->SetLineColor(3);
      pgLxyTrkEff_pp->SetMarkerColor(3);
      pgLxyTrkEff_pp->SetMarkerStyle(3);
      
      pgEtaTrkEff_pp->SetLineColor(3);
      pgEtaTrkEff_pp->SetMarkerColor(3);
      pgEtaTrkEff_pp->SetMarkerStyle(3);

      pgPtTrkEff_std->SetLineColor(4);
      pgPtTrkEff_std->SetMarkerColor(4);
      pgPtTrkEff_std->SetMarkerStyle(4);
      
      pgLxyTrkEff_std->SetLineColor(4);
      pgLxyTrkEff_std->SetMarkerColor(4);
      pgLxyTrkEff_std->SetMarkerStyle(4);
      
      pgEtaTrkEff_std->SetLineColor(4);
      pgEtaTrkEff_std->SetMarkerColor(4);
      pgEtaTrkEff_std->SetMarkerStyle(4);
      
      pgPtTrkEff_regit->SetLineColor(2);
      pgPtTrkEff_regit->SetMarkerColor(2);
      pgPtTrkEff_regit->SetMarkerStyle(25);
      
      pgLxyTrkEff_regit->SetLineColor(2);
      pgLxyTrkEff_regit->SetMarkerColor(2);
      pgLxyTrkEff_regit->SetMarkerStyle(25);
      
      pgEtaTrkEff_regit->SetLineColor(2);
      pgEtaTrkEff_regit->SetMarkerColor(2);
      pgEtaTrkEff_regit->SetMarkerStyle(25);
          
      TCanvas *pcTrkEff = new TCanvas("pcTrkEff","pcTrkEff",1500,600);
      pcTrkEff->Divide(3,1);
      pcTrkEff->cd(1); //gPad->SetLogy();
      phPtTmp2->Draw();
      pgPtTrkEff_pp->Draw("pl same");
      pgPtTrkEff_std->Draw("pl same");
      pgPtTrkEff_regit->Draw("pl same");
      
      pcTrkEff->cd(2); //gPad->SetLogy();
      phLxyTmp2->Draw();
      pgLxyTrkEff_pp->Draw("pl same");
      pgLxyTrkEff_std->Draw("pl same");
      pgLxyTrkEff_regit->Draw("pl same");
      
      pcTrkEff->cd(3); //gPad->SetLogy();
      phEtaTmp->Draw();
      pgEtaTrkEff_pp->Draw("pl same");
      pgEtaTrkEff_std->Draw("pl same");
      pgEtaTrkEff_regit->Draw("pl same");
      
      TLegend *t2 = new TLegend(0.34,0.78,0.94,0.91);
      t2->SetBorderSize(0);
      t2->SetFillStyle(0);
      t2->AddEntry(pgPtTrkEff_pp,aRecoNames[0],"pl");
      t2->AddEntry(pgPtTrkEff_std,aRecoNames[1],"pl");
      t2->AddEntry(pgPtTrkEff_regit,aRecoNames[2],"pl");
      t2->Draw();

      // ######################## eta plots
      if(bDoEta)
	{
	    // barrel
	  TH1D *phPtGenRecoTrk_pp_barrel   = new TH1D("phPtGenRecoTrk_pp_barrel","phPtGenRecoTrk_pp_barrel",nPtTrkBins,ptTrkMin,ptTrkMax);
	  TH1D *phEtaGenRecoTrk_pp_barrel  = new TH1D("phEtaGenRecoTrk_pp_barrel","phEtaGenRecoTrk_pp_barrel",12,-2.4,2.4);
	  phPtGenRecoTrk_pp_barrel->Sumw2();
	  phEtaGenRecoTrk_pp_barrel->Sumw2();
	  
	  TH1D *phPtGenRecoTrk_std_barrel   = new TH1D("phPtGenRecoTrk_std_barrel","phPtGenRecoTrk_std_barrel",nPtTrkBins,ptTrkMin,ptTrkMax);
	  TH1D *phEtaGenRecoTrk_std_barrel  = new TH1D("phEtaGenRecoTrk_std_barrel","phEtaGenRecoTrk_std_barrel",12,-2.4,2.4);
	  phPtGenRecoTrk_std_barrel->Sumw2();
	  phEtaGenRecoTrk_std_barrel->Sumw2();
	  
	  TH1D *phPtGenRecoTrk_regit_barrel = new TH1D("phPtGenRecoTrk_regit_barrel","phPtGenRecoTrk_regit_barrel",nPtTrkBins,ptTrkMin,ptTrkMax);
	  TH1D *phEtaGenRecoTrk_regit_barrel= new TH1D("phEtaGenRecoTrk_regit_barrel","phEtaGenRecoTrk_regit_barrel",12,-2.4,2.4);
	  phPtGenRecoTrk_regit_barrel->Sumw2();
	  phEtaGenRecoTrk_regit_barrel->Sumw2();
	  
	  // intermediate
	  TH1D *phPtGenRecoTrk_pp_inter   = new TH1D("phPtGenRecoTrk_pp_inter","phPtGenRecoTrk_pp_inter",nPtTrkBins,ptTrkMin,ptTrkMax);
	  TH1D *phEtaGenRecoTrk_pp_inter  = new TH1D("phEtaGenRecoTrk_pp_inter","phEtaGenRecoTrk_pp_inter",12,-2.4,2.4);
	  phPtGenRecoTrk_pp_inter->Sumw2();
	  phEtaGenRecoTrk_pp_inter->Sumw2();
	  
	  TH1D *phPtGenRecoTrk_std_inter   = new TH1D("phPtGenRecoTrk_std_inter","phPtGenRecoTrk_std_inter",nPtTrkBins,ptTrkMin,ptTrkMax);
	  TH1D *phEtaGenRecoTrk_std_inter  = new TH1D("phEtaGenRecoTrk_std_inter","phEtaGenRecoTrk_std_inter",12,-2.4,2.4);
	  phPtGenRecoTrk_std_inter->Sumw2();
	  phEtaGenRecoTrk_std_inter->Sumw2();
	  
	  TH1D *phPtGenRecoTrk_regit_inter = new TH1D("phPtGenRecoTrk_regit_inter","phPtGenRecoTrk_regit_inter",nPtTrkBins,ptTrkMin,ptTrkMax);
	  TH1D *phEtaGenRecoTrk_regit_inter= new TH1D("phEtaGenRecoTrk_regit_inter","phEtaGenRecoTrk_regit_inter",12,-2.4,2.4);
	  phPtGenRecoTrk_regit_inter->Sumw2();
	  phEtaGenRecoTrk_regit_inter->Sumw2();
	  
	  // fwd
	  TH1D *phPtGenRecoTrk_pp_fwd   = new TH1D("phPtGenRecoTrk_pp_fwd","phPtGenRecoTrk_pp_fwd",nPtTrkBins,ptTrkMin,ptTrkMax);
	  TH1D *phEtaGenRecoTrk_pp_fwd  = new TH1D("phEtaGenRecoTrk_pp_fwd","phEtaGenRecoTrk_pp_fwd",12,-2.4,2.4);
	  phPtGenRecoTrk_pp_fwd->Sumw2();
	  phEtaGenRecoTrk_pp_fwd->Sumw2();
	  
	  TH1D *phPtGenRecoTrk_std_fwd   = new TH1D("phPtGenRecoTrk_std_fwd","phPtGenRecoTrk_std_fwd",nPtTrkBins,ptTrkMin,ptTrkMax);
	  TH1D *phEtaGenRecoTrk_std_fwd  = new TH1D("phEtaGenRecoTrk_std_fwd","phEtaGenRecoTrk_std_fwd",12,-2.4,2.4);
	  phPtGenRecoTrk_std_fwd->Sumw2();
	  phEtaGenRecoTrk_std_fwd->Sumw2();
	  
	  TH1D *phPtGenRecoTrk_regit_fwd = new TH1D("phPtGenRecoTrk_regit_fwd","phPtGenRecoTrk_regit_fwd",nPtTrkBins,ptTrkMin,ptTrkMax);
	  TH1D *phEtaGenRecoTrk_regit_fwd= new TH1D("phEtaGenRecoTrk_regit_fwd","phEtaGenRecoTrk_regit_fwd",12,-2.4,2.4);
	  phPtGenRecoTrk_regit_fwd->Sumw2();
	  phEtaGenRecoTrk_regit_fwd->Sumw2();
	  
	  pcTemp2->cd();

     double nPp = ptTrkPp->GetEntries();
     double nStd = ptTrkStd->GetEntries();
     double nRegit = ptTrkRegit->GetEntries();

	  //barrel
     ptTrkPp->Draw("ptreco>>phPtGenRecoTrk_pp_barrel",trkRecoSelection_barrel,"Esame");
	  ptTrkStd->Draw("ptreco>>phPtGenRecoTrk_std_barrel",trkRecoSelection_barrel,"Esame");
	  ptTrkRegit->Draw("ptreco>>phPtGenRecoTrk_regit_barrel",trkRecoSelection_barrel,"Esame");
     phPtGenRecoTrk_pp_barrel->Scale(1./nPp);
     phPtGenRecoTrk_std_barrel->Scale(1./nStd);
     phPtGenRecoTrk_regit_barrel->Scale(1./nRegit);

	  //intermed
	  ptTrkPp->Draw("ptreco>>phPtGenRecoTrk_pp_inter",trkRecoSelection_inter,"Esame");
	  ptTrkStd->Draw("ptreco>>phPtGenRecoTrk_std_inter",trkRecoSelection_inter,"Esame");
	  ptTrkRegit->Draw("ptreco>>phPtGenRecoTrk_regit_inter",trkRecoSelection_inter,"Esame");
     phPtGenRecoTrk_pp_inter->Scale(1./nPp);
     phPtGenRecoTrk_std_inter->Scale(1./nStd);
     phPtGenRecoTrk_regit_inter->Scale(1./nRegit);

	  //fwd
	  ptTrkPp->Draw("ptreco>>phPtGenRecoTrk_pp_fwd",trkRecoSelection_fwd,"Esame");
	  ptTrkStd->Draw("ptreco>>phPtGenRecoTrk_std_fwd",trkRecoSelection_fwd,"Esame");
	  ptTrkRegit->Draw("ptreco>>phPtGenRecoTrk_regit_fwd",trkRecoSelection_fwd,"Esame");
     phPtGenRecoTrk_pp_fwd->Scale(1./nPp);
     phPtGenRecoTrk_std_fwd->Scale(1./nStd);
     phPtGenRecoTrk_regit_fwd->Scale(1./nRegit);
	  
	  // barrel
	  TH1D *pgPtTrkEff_barrel   = new TH1D("pgPtTrkEff_barrel","pgPtTrkEff_barrel",nPtTrkBins,ptTrkMin,ptTrkMax);
	  pgPtTrkEff_barrel->Sumw2();
	  TH1D *pgPtTrkEff2_barrel   = new TH1D("pgPtTrkEff2_barrel","pgPtTrkEff2_barrel",nPtTrkBins,ptTrkMin,ptTrkMax);
	  pgPtTrkEff2_barrel->Sumw2();
	  
	  //intermed
	  TH1D *pgPtTrkEff_inter   = new TH1D("pgPtTrkEff_inter","pgPtTrkEff_inter",nPtTrkBins,ptTrkMin,ptTrkMax);
	  pgPtTrkEff_inter->Sumw2();
	  TH1D *pgPtTrkEff2_inter   = new TH1D("pgPtTrkEff2_inter","pgPtTrkEff2_inter",nPtTrkBins,ptTrkMin,ptTrkMax);
	  pgPtTrkEff2_inter->Sumw2();

	  //fwd
	  TH1D *pgPtTrkEff_fwd   = new TH1D("pgPtTrkEff_fwd","pgPtTrkEff_fwd",nPtTrkBins,ptTrkMin,ptTrkMax);
	  pgPtTrkEff_fwd->Sumw2();
	  TH1D *pgPtTrkEff2_fwd   = new TH1D("pgPtTrkEff2_fwd","pgPtTrkEff2_fwd",nPtTrkBins,ptTrkMin,ptTrkMax);
	  pgPtTrkEff2_fwd->Sumw2();
		  
	  pgPtTrkEff_barrel->Divide(phPtGenRecoTrk_std_barrel,phPtGenRecoTrk_regit_barrel,1,1,"b");
	  pgPtTrkEff_inter->Divide(phPtGenRecoTrk_std_inter,phPtGenRecoTrk_regit_inter,1,1,"b");
	  pgPtTrkEff_fwd->Divide(phPtGenRecoTrk_std_fwd,phPtGenRecoTrk_regit_fwd,1,1,"b");
	  pgPtTrkEff2_barrel->Divide(phPtGenRecoTrk_regit_barrel,phPtGenRecoTrk_pp_barrel,1,1,"b");
	  pgPtTrkEff2_inter->Divide(phPtGenRecoTrk_regit_inter,phPtGenRecoTrk_pp_inter,1,1,"b");
	  pgPtTrkEff2_fwd->Divide(phPtGenRecoTrk_regit_fwd,phPtGenRecoTrk_pp_fwd,1,1,"b");

	  //drawing
	  // barrel
	  pgPtTrkEff_barrel->SetLineColor(kGreen+2);
	  pgPtTrkEff_barrel->SetLineStyle(2);
	  pgPtTrkEff_barrel->SetMarkerColor(kGreen+2);
	  pgPtTrkEff_barrel->SetMarkerStyle(33);
	  pgPtTrkEff2_barrel->SetLineColor(kGreen-2);
	  pgPtTrkEff2_barrel->SetLineStyle(2);
	  pgPtTrkEff2_barrel->SetMarkerColor(kGreen-2);
	  pgPtTrkEff2_barrel->SetMarkerStyle(33);
         	 
	  //intermediate
	  pgPtTrkEff_inter->SetLineColor(kBlue+2);
	  pgPtTrkEff_inter->SetLineStyle(2);
	  pgPtTrkEff_inter->SetMarkerColor(kBlue+2);
	  pgPtTrkEff_inter->SetMarkerStyle(21);
	  pgPtTrkEff2_inter->SetLineColor(kBlue-2);
	  pgPtTrkEff2_inter->SetLineStyle(2);
	  pgPtTrkEff2_inter->SetMarkerColor(kBlue-2);
	  pgPtTrkEff2_inter->SetMarkerStyle(21);
          
	  //forward
	  pgPtTrkEff_fwd->SetLineColor(kOrange+2);
	  pgPtTrkEff_fwd->SetLineStyle(2);
	  pgPtTrkEff_fwd->SetMarkerColor(kOrange+2);
	  pgPtTrkEff_fwd->SetMarkerStyle(20);
	  pgPtTrkEff2_fwd->SetLineColor(kOrange-2);
	  pgPtTrkEff2_fwd->SetLineStyle(2);
	  pgPtTrkEff2_fwd->SetMarkerColor(kOrange-2);
	  pgPtTrkEff2_fwd->SetMarkerStyle(20);
          	  
	  TLatex lx3;
	  TCanvas *pcTrkEtaEff = new TCanvas("pcTrkEtaEff","pcTrkEtaEff");
	  pcTrkEtaEff->cd(1); //gPad->SetLogy();
	  phPtTmp2->Draw();
	  pgPtTrkEff_barrel->Draw("pl same");
	  pgPtTrkEff_inter->Draw("pl same");
	  pgPtTrkEff_fwd->Draw("pl same");
     // lx3.DrawLatex(1,0.7,"Std_reco/Regit_reco");
	  lx3.DrawLatex(1,0.7,Form("%s / %s",aRecoNames[1],aRecoNames[2]));
	
	  TLegend *t3 = new TLegend(0.34,0.2,0.54,0.4);
	  t3->SetBorderSize(0);
	  t3->SetFillStyle(0);
	  t3->AddEntry(pgPtTrkEff_barrel,"|#eta^{#mu}|<1.2","pl");
	  t3->AddEntry(pgPtTrkEff_inter,"1.2<|#eta^{#mu}|<1.6","pl");
	  t3->AddEntry(pgPtTrkEff_fwd,"1.6<|#eta^{#mu}|<2.4","pl");
	  t3->Draw();
	  
	  TLatex lx32;
	  TCanvas *pcTrkEtaEff2 = new TCanvas("pcTrkEtaEff2","pcTrkEtaEff2");
	  pcTrkEtaEff2->cd(1); //gPad->SetLogy();
	  phPtTmp2->Draw();
	  pgPtTrkEff2_barrel->Draw("pl same");
	  pgPtTrkEff2_inter->Draw("pl same");
	  pgPtTrkEff2_fwd->Draw("pl same");
     // lx32.DrawLatex(1,0.7,"Regit_reco/Pp_reco");
	  lx3.DrawLatex(1,0.7,Form("%s / %s",aRecoNames[2],aRecoNames[0]));
	
	  TLegend *t32 = new TLegend(0.34,0.2,0.54,0.4);
	  t32->SetBorderSize(0);
	  t32->SetFillStyle(0);
	  t32->AddEntry(pgPtTrkEff2_barrel,"|#eta^{#mu}|<1.2","pl");
	  t32->AddEntry(pgPtTrkEff2_inter,"1.2<|#eta^{#mu}|<1.6","pl");
	  t32->AddEntry(pgPtTrkEff2_fwd,"1.6<|#eta^{#mu}|<2.4","pl");
	  t32->Draw();
	  
	  if(bSavePlots)
	    {
	      TString outFileBase1(Form("%s/%s_%s",pOutFilePath,pcTrkEtaEff->GetTitle(),pOutFileName));
	      TString outFileGif1 = outFileBase1 + ".gif";
	      pcTrkEtaEff->Print(outFileGif1.Data(),"gifLandscape");
	      TString outFilePdf1 = outFileBase1 + ".pdf";
	      pcTrkEtaEff->SaveAs(outFilePdf1.Data());
	      TString outFileBase12(Form("%s/%s_%s",pOutFilePath,pcTrkEtaEff2->GetTitle(),pOutFileName));
	      TString outFileGif12 = outFileBase12 + ".gif";
	      pcTrkEtaEff2->Print(outFileGif12.Data(),"gifLandscape");
	      TString outFilePdf12 = outFileBase12 + ".pdf";
	      pcTrkEtaEff2->SaveAs(outFilePdf12.Data());
	      
	    }
	}

      if(bSavePlots)
	{
	  TString outFileBase(Form("%s/%s_%s",pOutFilePath,pcTrkEff->GetTitle(),pOutFileName));
	  TString outFileGif = outFileBase+".gif";
	  pcTrkEff->Print(outFileGif.Data(),"gifLandscape");
	  TString outFilePdf = outFileBase+".pdf";
	  pcTrkEff->Print(outFilePdf.Data());
	}
    }


}


//____________________________________________________________________________________
