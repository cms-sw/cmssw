#ifndef RecoLocalMuon_Histograms_H
#define RecoLocalMuon_Histograms_H

/** \class Histograms
 *  Collection of histograms for DT RecHit and Segment test.
 *
 *  $Date: 2006/03/24 11:09:57 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */


#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"

#include <string>


//---------------------------------------------------------------------------------------
/// A set of histograms of residuals and pulls for 1D RecHits
class HRes1DHit{
 public:
  HRes1DHit(std::string name_){
    TString N = name_.c_str();
    name=N;

    // Position, sigma, residual, pull
    hDist       = new TH1F ("1D_"+N+"_hDist", "1D RHit distance from wire", 100, 0,2.5);
    hRes        = new TH1F ("1D_"+N+"_hRes", "1D RHit residual", 1720, -4.3,4.3);
    hResVsEta   = new TH2F("1D_"+N+"_hResVsEta", "1D RHit residual vs eta",
			   50, -1.25,1.25, 860, -4.3,4.3);
    hResVsPhi   = new TH2F("1D_"+N+"_hResVsPhi", "1D RHit residual vs phi",
			   100, -3.2, 3.2, 860, -4.3,4.3);
    hResVsPos   = new TH2F("1D_"+N+"_hResVsPos", "1D RHit residual vs position",
			   100, 0,2.5, 860, -4.3,4.3);    
    hPull       = new TH1F ("1D_"+N+"_hPull", "1D RHit pull", 100, -5,5);
  }
  
  HRes1DHit(TString name_, TFile* file){
    name=name_;
    hDist          = (TH1F *) file->Get("1D_"+name+"_hDist");
    hRes           = (TH1F *) file->Get("1D_"+name+"_hRes");
    hResVsEta      = (TH2F *) file->Get("1D_"+name+"_hResVsEta");
    hResVsPhi      = (TH2F *) file->Get("1D_"+name+"_hResVsPhi");
    hResVsPos      = (TH2F *) file->Get("1D_"+name+"_hResVsPos");
    hPull          = (TH1F *) file->Get("1D_"+name+"_hPull");
  }


  ~HRes1DHit(){
//     delete hDist;
//     delete hRes;
//     delete hResVsEta;
//     delete hResVsPhi;
//     delete hResVsPos;
//     delete hPull;
  }

  void Fill(float distSimHit,
	    float distRecHit,
	    float etaSimHit,
	    float phiSimHit,
	    float errRecHit) {
    // Reso, pull
    hDist->Fill(distRecHit);
    float res = distRecHit-distSimHit;
    hRes->Fill(res);
    hResVsEta->Fill(etaSimHit,res);
    hResVsPhi->Fill(phiSimHit,res);
    hResVsPos->Fill(distSimHit,res);
    if(errRecHit!=0) {
      hPull->Fill(res/errRecHit);
    }
    else std::cout<<"Error: RecHit error = 0" << std::endl;
  }
  
  void Write() {
    hDist->Write();     
    hRes->Write();      
    hResVsEta->Write();   
    hResVsPhi->Write(); 
    hResVsPos->Write(); 
    hPull->Write();
  }

  
 public:
  TH1F* hDist;
  TH1F* hRes;
  TH2F* hResVsEta;
  TH2F* hResVsPhi;
  TH2F* hResVsPos;
  TH1F* hPull;
 
  TString name;

};


//---------------------------------------------------------------------------------------
/// A set of histograms for efficiency 1D DT RecHits
class HEff1DHit{
 public:
  HEff1DHit(std::string name_){
    TString N = name_.c_str();
    name=N;

    hEtaMuSimHit       = new TH1F("1D_"+N+"_hEtaMuSimHit", "SimHit Eta distribution",
				  100, -1.5, 1.5);
    hEtaRecHit         = new TH1F("1D_"+N+"_hEtaRecHit", "SimHit Eta distribution with 1D RecHit",
				  100, -1.5, 1.5);
    hEffVsEta = 0;


    hPhiMuSimHit       = new TH1F("1D_"+N+"_hPhiMuSimHit", "SimHit Phi distribution",
				  100, -TMath::Pi(),TMath::Pi());
    hPhiRecHit         = new TH1F("1D_"+N+"_hPhiRecHit", "SimHit Phi distribution with 1D RecHit",
				  100, -TMath::Pi(),TMath::Pi());
    hEffVsPhi = 0;


    hDistMuSimHit       = new TH1F("1D_"+N+"_hDistMuSimHit", "SimHit Distance from wire distribution",
				   100, 0, 2.5);
    hDistRecHit         = new TH1F("1D_"+N+"_hDistRecHit", "SimHit Distance from wire distribution with 1D RecHit",
				   100, 0, 2.5);
    hEffVsDist = 0;

  }
  
  HEff1DHit (TString name_, TFile* file){
    name=name_;
    hEtaMuSimHit        = (TH1F *) file->Get("1D_"+name+"_hEtaMuSimHit");
    hEtaRecHit          = (TH1F *) file->Get("1D_"+name+"_hEtaRecHit");
    hEffVsEta           = (TH1F *) file->Get("1D_"+name+"_hEffVsEta");

    hPhiMuSimHit        = (TH1F *) file->Get("1D_"+name+"_hPhiMuSimHit");
    hPhiRecHit          = (TH1F *) file->Get("1D_"+name+"_hPhiRecHit");
    hEffVsPhi           = (TH1F *) file->Get("1D_"+name+"_hEffVsPhi");

    hDistMuSimHit       = (TH1F *) file->Get("1D_"+name+"_hDistMuSimHit");
    hDistRecHit         = (TH1F *) file->Get("1D_"+name+"_hDistRecHit");
    hEffVsDist          = (TH1F *) file->Get("1D_"+name+"_hEffVsDist");
  }


  ~HEff1DHit(){

//     delete hEtaMuSimHit;
//     delete hEtaRecHit;
//     if(hEffVsEta != 0)
//       delete hEffVsEta;

//     delete hPhiMuSimHit;
//     delete hPhiRecHit;
//     if(hEffVsPhi != 0)
//       delete hEffVsPhi;

//     delete hDistMuSimHit;
//     delete hDistRecHit;
//     if(hEffVsDist != 0)
//       delete hEffVsDist;

  }

  void Fill(float distSimHit,
	    float etaSimHit,
	    float phiSimHit,
	    bool fillRecHit) {

    hEtaMuSimHit->Fill(etaSimHit);
    hPhiMuSimHit->Fill(phiSimHit);
    hDistMuSimHit->Fill(distSimHit);
    if(fillRecHit) {
      hEtaRecHit->Fill(etaSimHit);
      hPhiRecHit->Fill(phiSimHit);
      hDistRecHit->Fill(distSimHit);
    }
  }
  


  void ComputeEfficiency() {

    hEffVsEta = (TH1F *) hEtaRecHit->Clone();
    hEffVsEta->SetName("1D_"+name+"_hEffVsEta");
    hEffVsEta->SetTitle("1D RecHit Efficiency as a function of Eta");
    hEffVsEta->Divide(hEtaMuSimHit);
    // Set the error accordingly to binomial statistics
    int nBinsEta = hEffVsEta->GetNbinsX();
    for(int bin = 1; bin <=  nBinsEta; bin++) {
      float nSimHit = hEtaMuSimHit->GetBinContent(bin);
      float eff = hEffVsEta->GetBinContent(bin);
      float error = 0;
      if(nSimHit != 0) {
        error = sqrt(eff*(1-eff)/nSimHit);
      }
      hEffVsEta->SetBinError(bin, error);
    }

    hEffVsPhi = (TH1F *) hPhiRecHit->Clone();
    hEffVsPhi->SetName("1D_"+name+"_hEffVsPhi");
    hEffVsPhi->SetTitle("1D RecHit Efficiency as a function of Phi");
    hEffVsPhi->Divide(hPhiMuSimHit);
    // Set the error accordingly to binomial statistics
    int nBinsPhi = hEffVsPhi->GetNbinsX();
    for(int bin = 1; bin <=  nBinsPhi; bin++) {
      float nSimHit = hPhiMuSimHit->GetBinContent(bin);
      float eff = hEffVsPhi->GetBinContent(bin);
      float error = 0;
      if(nSimHit != 0) {
        error = sqrt(eff*(1-eff)/nSimHit);
      }
      hEffVsPhi->SetBinError(bin, error);
    }

    hEffVsDist = (TH1F *) hDistRecHit->Clone();
    hEffVsDist->SetName("1D_"+name+"_hEffVsDist");
    hEffVsDist->SetTitle("1D RecHit Efficiency as a function of Dist");
    hEffVsDist->Divide(hDistMuSimHit);
    // Set the error accordingly to binomial statistics
    int nBinsDist = hEffVsDist->GetNbinsX();
    for(int bin = 1; bin <=  nBinsDist; bin++) {
      float nSimHit = hDistMuSimHit->GetBinContent(bin);
      float eff = hEffVsDist->GetBinContent(bin);
      float error = 0;
      if(nSimHit != 0) {
        error = sqrt(eff*(1-eff)/nSimHit);
      }
      hEffVsDist->SetBinError(bin, error);
    }


  }

  void Write() {
    hEtaMuSimHit->Write();                         
    hEtaRecHit->Write();
    if(hEffVsEta != 0)
      hEffVsEta->Write();                    
    hPhiMuSimHit->Write();                         
    hPhiRecHit->Write();                         
    if(hEffVsPhi != 0)
      hEffVsPhi->Write();               
    hDistMuSimHit->Write();                         
    hDistRecHit->Write();                         
    if(hEffVsDist != 0)
      hEffVsDist->Write();
  }

 public:
  TH1F* hEtaMuSimHit;
  TH1F* hEtaRecHit;
  TH1F* hEffVsEta;

  TH1F* hPhiMuSimHit;
  TH1F* hPhiRecHit;
  TH1F* hEffVsPhi;
                  
  TH1F* hDistMuSimHit;
  TH1F* hDistRecHit;
  TH1F* hEffVsDist;

  TString name;

};


#endif
