#ifndef RecoLocalMuon_Histograms_H
#define RecoLocalMuon_Histograms_H

/** \class Histograms
 *  Collection of histograms for DT RecHit and Segment test.
 *
 *  $Date: 2010/09/17 10:58:41 $
 *  $Revision: 1.7 $
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */


#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"
#include "TMath.h"

#include <string>
#include <iostream>
#include <math.h>

//---------------------------------------------------------------------------------------
/// A set of histograms of residuals and pulls for 1D RecHits
class HRes1DHit{
  public:
    HRes1DHit(std::string name_){
      TString N = name_.c_str();
      name=N;
      cout << "constructor Histo"<<endl;
      // Position, sigma, residual, pull
      hDist       = new TH1F ("1D_"+N+"_hDist", "1D RHit distance from wire", 100, 0,2.5);
      hRes        = new TH1F ("1D_"+N+"_hRes", "1D RHit residual", 300, -1.5,1.5);
      hResSt[0] = new TH1F("1D_"+N + "_hResMB1","1D RHit residual", 300, -0.5,0.5);
      hResSt[1] = new TH1F("1D_"+N + "_hResMB2","1D RHit residual", 300, -0.5,0.5);
      hResSt[2] = new TH1F("1D_"+N + "_hResMB3","1D RHit residual", 300, -0.5,0.5);
      hResSt[3] = new TH1F("1D_"+N + "_hResMB4","1D RHit residual", 300, -0.5,0.5);

      hResVsEta   = new TH2F("1D_"+N+"_hResVsEta", "1D RHit residual vs eta",
                             50, -1.25,1.25, 150, -1.5,1.5);
      hResVsPhi   = new TH2F("1D_"+N+"_hResVsPhi", "1D RHit residual vs phi",
                             100, -3.2, 3.2, 150, -1.5,1.5);
      hResVsPos   = new TH2F("1D_"+N+"_hResVsPos", "1D RHit residual vs position",
                             100, 0, 2.5, 150, -1.5,1.5);    
      hResVsAngle   = new TH2F("1D_"+N+"_hResVsAngle", "1D RHit residual vs impact angle",
                               100, 0.,1.2, 150, -1.5,1.5);    
      hResVsDistFE = new TH2F("1D_"+N+"_hResVsDistFE", "1D RHit residual vs FE distance",
                              100, 0.,400., 150, -1.5,1.5);    
      hPull       = new TH1F ("1D_"+N+"_hPull", "1D RHit pull", 100, -5,5);
      hPullSt[0] = new TH1F("1D_"+N + "_hPullMB1","1D RHit residual", 300, -5,5);
      hPullSt[1] = new TH1F("1D_"+N + "_hPullMB2","1D RHit residual", 300, -5,5);
      hPullSt[2] = new TH1F("1D_"+N + "_hPullMB3","1D RHit residual", 300, -5,5);
      hPullSt[3] = new TH1F("1D_"+N + "_hPullMB4","1D RHit residual", 300, -5,5);
      hPullVsPos  = new TH2F ("1D_"+N+"_hPullVsPos", "1D RHit pull vs position", 100, 0,2.5, 100, -5,5);
      hPullVsAngle  = new TH2F ("1D_"+N+"_hPullVsAngle", "1D RHit pull vs impact angle",
                                100, 0.,+1.2, 100, -5,5);
      hPullVsDistFE  = new TH2F ("1D_"+N+"_hPullVsDistFE", "1D RHit pull vs FE distance",100, 0., 400., 100, -5,5);
       hPullVsEta   = new TH2F("1D_"+N+"_hPullVsEta", "1D RHit residual vs eta",
                             50, -1.25,1.25, 150, -5,5);
      hPullVsPhi   = new TH2F("1D_"+N+"_hResVsPhi", "1D RHit residual vs phi",
                             100, -3.2, 3.2, 150, -5,5);
                                
    }

    HRes1DHit(TString name_, TFile* file){
      name=name_;
      hDist          = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hDist");
      hRes           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hRes");
      hResSt[0]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hResMB1");
      hResSt[1]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hResMB2");
      hResSt[2]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hResMB3");
      hResSt[3]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hResMB4");
      hResVsEta      = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hResVsEta");
      hResVsPhi      = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hResVsPhi");
      hResVsPos      = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hResVsPos");
      hResVsAngle    = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hResVsAngle");
      hResVsDistFE   = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hResVsDistFE");
      hPull          = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hPull");
      hPullSt[0]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hPullMB1");
      hPullSt[1]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hPullMB2");
      hPullSt[2]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hPullMB3");
      hPullSt[3]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hPullMB4");
      hPullVsPos     = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hPullVsPos");
      hPullVsAngle   = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hPullVsAngle");
      hPullVsDistFE  = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hPullVsDistFE");
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
              float thetaSimHit,
              float distFESimHit,
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
      hResVsAngle->Fill(thetaSimHit,res);
      hResVsDistFE->Fill(distFESimHit,res);
      if(errRecHit!=0) {
        float pull=res/errRecHit;
        hPull->Fill(pull);
        hPullVsPos->Fill(distSimHit,pull);
        hPullVsAngle->Fill(thetaSimHit,pull);
        hPullVsDistFE->Fill(distFESimHit,pull);
      }
      else std::cout<<"Error: RecHit error = 0" << std::endl;
    }

    void Write() {
      hDist->Write();     
      hRes->Write();      
      hResSt[0]->Write();      
      hResSt[1]->Write();      
      hResSt[2]->Write();      
      hResSt[3]->Write();      
      hResVsEta->Write();   
      hResVsPhi->Write(); 
      hResVsPos->Write(); 
      hResVsAngle->Write(); 
      hResVsDistFE->Write(); 
      hPull->Write();
      hPullSt[0]->Write();      
      hPullSt[1]->Write();      
      hPullSt[2]->Write();      
      hPullSt[3]->Write();      
      hPullVsPos->Write();
      hPullVsPhi->Write(); 
      hPullVsPos->Write(); 
      hPullVsAngle->Write();
      hPullVsDistFE->Write();
    }


  public:
    TH1F* hDist;
    TH1F* hRes;
    TH1F* hResSt[4];
    TH2F* hResVsEta;
    TH2F* hResVsPhi;
    TH2F* hResVsPos;
    TH2F* hResVsAngle;
    TH2F* hResVsDistFE;

    TH1F* hPull;
    TH1F* hPullSt[4];
    TH2F* hPullVsPos;
    TH2F* hPullVsAngle;
    TH2F* hPullVsDistFE;

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
      hEtaMuSimHit        = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hEtaMuSimHit");
      hEtaRecHit          = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hEtaRecHit");
      hEffVsEta           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hEffVsEta");

      hPhiMuSimHit        = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hPhiMuSimHit");
      hPhiRecHit          = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hPhiRecHit");
      hEffVsPhi           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hEffVsPhi");

      hDistMuSimHit       = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hDistMuSimHit");
      hDistRecHit         = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hDistRecHit");
      hEffVsDist          = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/1D_"+name+"_hEffVsDist");
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

//---------------------------------------------------------------//

// Histos of residuals for 2D rechits
class HRes2DHit{
  public:
    HRes2DHit(std::string name_){
      TString N = name_.c_str();
      name=N;

      hRecAngle = new TH1F ("2D_"+N+"_hRecAngle", "Distribution of Rec segment angles;angle (rad)",
                            100, -3.5, 3.5);
      hSimAngle = new TH1F ("2D_"+N+"_hSimAngle", "Distribution of segment angles from SimHits;angle (rad)",
                            100, -3.5, 3.5);
      hRecVsSimAngle = new TH2F ("2D_"+N+"_hRecVsSimAngle", "Rec angle vs sim angle;angle (rad)",
                                 100, -3.5, 3.5, 100, -3.5, 3.5);


      hResAngle   = new TH1F ("2D_"+N+"_hResAngle", "Residual on 2D segment angle;angle_{rec}-angle_{sim} (rad)", 150, -0.15, 0.15);
      hResAngleVsEta   = new TH2F ("2D_"+N+"_hResAngleVsEta", "Residual on 2D segment angle vs Eta; #eta; res (rad)",
                                   100, -2.5, 2.5, 200, -0.2, 0.2);
      hResAngleVsPhi   = new TH2F ("2D_"+N+"_hResAngleVsPhi", "Residual on 2D segment angle vs Phi; #phi (rad);res (rad)",
                                   100, -3.2, 3.2, 150, -0.2, 0.2);

      hResPos   = new TH1F ("2D_"+N+"_hResPos", "Residual on 2D segment position (x at SL center);x_{rec}-x_{sim} (cm)",
                            150, -0.2, 0.2);
      hResPosVsEta   = new TH2F ("2D_"+N+"_hResPosVsEta", "Residual on 2D segment position vs Eta;#eta;res (cm)",
                                 100, -2.5, 2.5, 150, -0.2, 0.2);
      hResPosVsPhi   = new TH2F ("2D_"+N+"_hResPosVsPhi", "Residual on 2D segment position vs Phi;#phi (rad);res (cm)",
                                 100, -3.2, 3.2, 150, -0.2, 0.2);

      hResPosVsResAngle = new TH2F("2D_"+N+"_hResPosVsResAngle",
                                   "Residual on 2D segment position vs Residual on 2D segment angle;angle (rad);res (cm)",
                                   100, -0.3, 0.3, 150, -0.2, 0.2);

      hPullAngle   = new TH1F ("2D_"+N+"_hPullAngle", "Pull on 2D segment angle;(angle_{rec}-angle_{sim})/#sigma (rad)", 150, -5, 5);
      hPullPos   = new TH1F ("2D_"+N+"_hPullPos", "Pull on 2D segment position (x at SL center);(x_{rec}-x_{sim} (cm))/#sigma",
                             150, -5, 5);
    }

    HRes2DHit (TString name_, TFile* file){
      name=name_;

      hRecAngle = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hRecAngle");
      hSimAngle = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hSimAngle");
      hRecVsSimAngle = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hRecVsSimAngle");
      hResAngle = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hResAngle");
      hResAngleVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hResAngleVsEta");
      hResAngleVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hResAngleVsPhi");
      hResPos = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hResPos");
      hResPosVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hResPosVsEta");
      hResPosVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hResPosVsPhi");
      hResPosVsResAngle = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hResPosVsResAngle");
      hPullAngle = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hPullAngle");
      hPullPos = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hPullPos");

    }


    ~HRes2DHit(){
    }

    void Fill(float angleSimSegment,
              float angleRecSegment,
              float posSimSegment,
              float posRecSegment,
              float etaSimSegment,
              float phiSimSegment,
              float sigmaPos,
              float sigmaAngle) {

      hRecAngle->Fill(angleRecSegment);
      hSimAngle->Fill(angleSimSegment);
      hRecVsSimAngle->Fill(angleSimSegment, angleRecSegment);
      float resAngle = angleRecSegment-angleSimSegment;
      hResAngle->Fill(resAngle);
      hResAngleVsEta->Fill(etaSimSegment, resAngle);
      hResAngleVsPhi->Fill(phiSimSegment, resAngle);
      float resPos = posRecSegment-posSimSegment;
      hResPos->Fill(resPos);
      hResPosVsEta->Fill(etaSimSegment, resPos);
      hResPosVsPhi->Fill(phiSimSegment, resPos);
      hResPosVsResAngle->Fill(resAngle, resPos);
      hPullAngle->Fill(resAngle/sigmaAngle);
      hPullPos->Fill(resPos/sigmaPos);
    }

    void Write() {

      hRecAngle->Write();
      hSimAngle->Write();
      hRecVsSimAngle->Write();
      hResAngle->Write();
      hResAngleVsEta->Write();
      hResAngleVsPhi->Write();
      hResPos->Write();
      hResPosVsEta->Write();
      hResPosVsPhi->Write();
      hResPosVsResAngle->Write();
      hPullAngle->Write();
      hPullPos->Write();
    }


  public:
    TH1F *hRecAngle;
    TH1F *hSimAngle;
    TH2F *hRecVsSimAngle;
    TH1F *hResAngle;
    TH2F *hResAngleVsEta;
    TH2F *hResAngleVsPhi;
    TH1F *hResPos;
    TH2F *hResPosVsEta;
    TH2F *hResPosVsPhi;
    TH2F *hResPosVsResAngle;
    TH1F *hPullAngle;
    TH1F *hPullPos;
    TString name;

};

//--------------------------------------------------------------------------------//

// Histos for 2D RecHit efficiency
class HEff2DHit{
  public:
    HEff2DHit(std::string name_){
      TString N = name_.c_str();
      name=N;

      hEtaSimSegm     = new TH1F("2D_"+N+"_hEtaSimSegm", "Eta of SimHit segment", 100, -1.5, 1.5);
      hEtaRecHit      = new TH1F("2D_"+N+"_hEtaRecHit", "Eta distribution of SimHit segment with 2D RecHit",
                                 100, -1.5, 1.5);
      hEffVsEta       = 0;

      hPhiSimSegm     = new TH1F("2D_"+N+"_hPhiSimSegm", "Phi of SimHit segment",
                                 100, -TMath::Pi(),TMath::Pi());
      hPhiRecHit      = new TH1F("2D_"+N+"_hPhiRecHit", "Phi distribution of SimHit segment with 2D RecHit",
                                 100, -TMath::Pi(),TMath::Pi());
      hEffVsPhi       = 0;


      hPosSimSegm     = new TH1F("2D_"+N+"_hPosSimSegm", "Position in SL of SimHit segment (cm)",
                                 100, -250, 250);
      hPosRecHit      = new TH1F("2D_"+N+"_hPosRecHit", "Position in SL of SimHit segment with 2D RecHit (cm)",
                                 100, -250, 250);
      hEffVsPos       = 0;


      hAngleSimSegm   = new TH1F("2D_"+N+"_hAngleSimSegm", "Angle of SimHit segment (rad)",
                                 100, -2, 2);
      hAngleRecHit    = new TH1F("2D_"+N+"_hAngleRecHit", "Angle of SimHit segment with 2D RecHit (rad)",
                                 100, -2, 2);
      hEffVsAngle     = 0;

    }

    HEff2DHit (TString name_, TFile* file){
      name=name_;
      hEtaSimSegm = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hEtaSimSegm");
      hEtaRecHit = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hEtaRecHit");
      hEffVsEta = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hEffVsEta");

      hPhiSimSegm = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hPhiSimSegm");
      hPhiRecHit = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hPhiRecHit");
      hEffVsPhi = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hEffVsPhi");

      hPosSimSegm = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hPosSimSegm");
      hPosRecHit = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hPosRecHit");
      hEffVsPos = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hEffVsPos");

      hAngleSimSegm = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hAngleSimSegm");
      hAngleRecHit = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hAngleRecHit");
      hEffVsAngle = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/2DSegments/2D_"+name+"_hEffVsAngle");
    }


    ~HEff2DHit(){
      //delete hEtaSimSegm;    
      //delete hEtaRecHit;     
      //if(hEffVsEta != 0)
      // delete hEffVsEta;   

      //delete hPhiSimSegm;    
      //delete hPhiRecHit;
      //if(hEffVsPhi != 0)
      //  delete hEffVsPhi;      

      //delete hPosSimSegm;    
      //delete hPosRecHit;
      //if(hEffVsPos != 0)
      //  delete hEffVsPos;      

      //delete hAngleSimSegm;  
      //delete hAngleRecHit;
      //if(hEffVsAngle != 0)
      //  delete hEffVsAngle;    
    }

    void Fill(float etaSimSegm,
              float phiSimSegm,
              float posSimSegm,
              float angleSimSegm,
              bool fillRecHit) {

      hEtaSimSegm->Fill(etaSimSegm);       
      hPhiSimSegm->Fill(phiSimSegm);   
      hPosSimSegm->Fill(posSimSegm);   
      hAngleSimSegm->Fill(angleSimSegm); 

      if(fillRecHit) {
        hEtaRecHit->Fill(etaSimSegm);    
        hPhiRecHit->Fill(phiSimSegm);    
        hPosRecHit->Fill(posSimSegm);    
        hAngleRecHit->Fill(angleSimSegm);  
      }
    }



    void ComputeEfficiency() {

      hEffVsEta = (TH1F *) hEtaRecHit->Clone();
      hEffVsEta->SetName("2D_"+name+"_hEffVsEta");
      hEffVsEta->SetTitle("2D RecHit Efficiency as a function of Eta");
      hEffVsEta->Divide(hEtaSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsEta = hEffVsEta->GetNbinsX();
      for(int bin = 1; bin <=  nBinsEta; bin++) {
        float nSimHit = hEtaSimSegm->GetBinContent(bin);
        float eff = hEffVsEta->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsEta->SetBinError(bin, error);
      }

      hEffVsPhi = (TH1F *) hPhiRecHit->Clone();
      hEffVsPhi->SetName("2D_"+name+"_hEffVsPhi");
      hEffVsPhi->SetTitle("2D RecHit Efficiency as a function of Phi");
      hEffVsPhi->Divide(hPhiSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsPhi = hEffVsPhi->GetNbinsX();
      for(int bin = 1; bin <=  nBinsPhi; bin++) {
        float nSimHit = hPhiSimSegm->GetBinContent(bin);
        float eff = hEffVsPhi->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsPhi->SetBinError(bin, error);
      }

      hEffVsPos = (TH1F *) hPosRecHit->Clone();
      hEffVsPos->SetName("2D_"+name+"_hEffVsPos");
      hEffVsPos->SetTitle("2D RecHit Efficiency as a function of position in SL");
      hEffVsPos->Divide(hPosSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsPos = hEffVsPos->GetNbinsX();
      for(int bin = 1; bin <=  nBinsPos; bin++) {
        float nSimHit = hPosSimSegm->GetBinContent(bin);
        float eff = hEffVsPos->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsPos->SetBinError(bin, error);
      }

      hEffVsAngle = (TH1F *) hAngleRecHit->Clone();
      hEffVsAngle->SetName("2D_"+name+"_hEffVsAngle");
      hEffVsAngle->SetTitle("2D RecHit Efficiency as a function of angle");
      hEffVsAngle->Divide(hAngleSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsAngle = hEffVsAngle->GetNbinsX();
      for(int bin = 1; bin <=  nBinsAngle; bin++) {
        float nSimHit = hAngleSimSegm->GetBinContent(bin);
        float eff = hEffVsAngle->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsAngle->SetBinError(bin, error);
      }

    }

    void Write() {
      hEtaSimSegm->Write();   
      hEtaRecHit->Write();    
      if(hEffVsEta != 0)
        hEffVsEta->Write();     
      hPhiSimSegm->Write();   
      hPhiRecHit->Write();    
      if(hEffVsPhi != 0)
        hEffVsPhi->Write();     
      hPosSimSegm->Write();   
      hPosRecHit->Write();    
      if(hEffVsPos != 0)
        hEffVsPos->Write();     
      hAngleSimSegm->Write(); 
      hAngleRecHit->Write();  
      if(hEffVsAngle != 0)
        hEffVsAngle->Write();   

    }

  public:

    TH1F *hEtaSimSegm;   
    TH1F *hEtaRecHit;    
    TH1F *hEffVsEta;     
    TH1F *hPhiSimSegm;   
    TH1F *hPhiRecHit;    
    TH1F *hEffVsPhi;     
    TH1F *hPosSimSegm;   
    TH1F *hPosRecHit;    
    TH1F *hEffVsPos;     
    TH1F *hAngleSimSegm; 
    TH1F *hAngleRecHit;  
    TH1F *hEffVsAngle;   

    TString name;

};
//---------------------------------------------------------------------------------------
// Histos of residuals for 4D rechits
class HRes4DHit{
  public:
    HRes4DHit(std::string name_){
      TString N = name_.c_str();
      name=N;

      hRecAlpha       = new TH1F ("4D_"+N+"_hRecAlpha", "4D RecHit alpha (RPhi) distribution;#alpha^{x} (rad)", 100, -3.5, 3.5);
      hRecBeta        = new TH1F ("4D_"+N+"_hRecBeta", "4D RecHit beta distribution:#alpha^{y} (rad)", 100, -3.5, 3.5);

      hSimAlpha       = new TH1F("4D_"+N+"_hSimAlpha", "4D segment from SimHit alpha (RPhi) distribution;i#alpha^{x} (rad)",
                                 100, -3.5, 3.5);
      hSimBeta        = new TH1F("4D_"+N+"_hSimBeta", "4D segment from SimHit beta distribution;#alpha^{y} (rad)",
                                 100, -3.5, 3.5);
      hRecVsSimAlpha  = new TH2F("4D_"+N+"_hRecVsSimAlpha", "4D segment rec alpha {v}s sim alpha (RPhi);#alpha^{x} (rad)",
                                 100, -3.5, 3.5, 100, -3.5, 3.5);
      hRecVsSimBeta   = new TH2F("4D_"+N+"_hRecVsSimBeta", "4D segment rec beta vs sim beta (RZ);#alpha^{y} (rad)",
                                 100, -3.5, 3.5, 100, -3.5, 3.5);

      hResAlpha       = new TH1F ("4D_"+N+"_hResAlpha", 
                                  "4D RecHit residual on #alpha_x direction;#alpha^{x}_{rec}-#alpha^{x}_{sim} (rad)",
                                  200, -0.015, 0.015);
      hResAlphaVsEta  = new TH2F ("4D_"+N+"_hResAlphaVsEta",
                                  "4D RecHit residual on #alpha_x direction vs eta;#eta;#alpha^{x}_{rec}-#alpha^{x}_{sim} (rad)",
                                  100, -2.5, 2.5, 100, -0.025, 0.025);
      hResAlphaVsPhi  = new TH2F ("4D_"+N+"_hResAlphaVsPhi",
                                  "4D RecHit residual on #alpha_x direction vs phi (rad);#phi (rad);#alpha^{x}_{rec}-#alpha^{x}_{sim} (rad)",
                                  100, -3.2, 3.2, 100, -0.025, 0.025);

      hResBeta        = new TH1F ("4D_"+N+"_hResBeta",
                                  "4D RecHit residual on beta direction;#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
                                  200, -0.1, 0.1);
      hResBetaVsEta   = new TH2F ("4D_"+N+"_hResBetaVsEta",
                                  "4D RecHit residual on beta direction vs eta;#eta;#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
                                  100, -2.5, 2.5, 200, -0.2, 0.2);
      hResBetaVsPhi   = new TH2F ("4D_"+N+"_hResBetaVsPhi",
                                  "4D RecHit residual on beta direction vs phi;#phi (rad);#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
                                  100, -3.2, 3.2, 200, -0.2, 0.2);

      hResX           = new TH1F ("4D_"+N+"_hResX", "4D RecHit residual on position (x) in chamber;x_{rec}-x_{sim} (cm)",
                                  150, -0.15, 0.15);
      hResXVsEta      = new TH2F ("4D_"+N+"_hResXVsEta", "4D RecHit residual on position (x) in chamber vs eta;#eta;x_{rec}-x_{sim} (cm)",
                                  100, -2.5, 2.5, 150, -0.3, 0.3);
      hResXVsPhi      = new TH2F ("4D_"+N+"_hResXVsPhi", "4D RecHit residual on position (x) in chamber vs phi;#phi (rad);x_{rec}-x_{sim} (cm)",
                                  100, -3.2, 3.2, 150, -0.3, 0.3);

      hResY           = new TH1F ("4D_"+N+"_hResY", "4D RecHit residual on position (y) in chamber;y_{rec}-y_{sim} (cm)", 150, -0.6, 0.6);
      hResYVsEta      = new TH2F ("4D_"+N+"_hResYVsEta", "4D RecHit residual on position (y) in chamber vs eta;#eta;y_{rec}-y_{sim} (cm)",
                                  100, -2.5, 2.5, 150, -0.6, 0.6);
      hResYVsPhi      = new TH2F ("4D_"+N+"_hResYVsPhi", "4D RecHit residual on position (y) in chamber vs phi;#phi (rad);y_{rec}-y_{sim} (cm)",
                                  100, -3.2, 3.2, 150, -0.6, 0.6);

      hResAlphaVsResBeta = new TH2F("4D_"+N+"_hResAlphaVsResBeta", "4D RecHit residual on alpha vs residual on beta",
                                    200, -0.3, 0.3, 500, -0.15, 0.15);
      hResXVsResY = new TH2F("4D_"+N+"_hResXVsResY", "4D RecHit residual on X vs residual on Y",
                             150, -0.6, 0.6, 50, -0.3, 0.3);
      hResAlphaVsResX = new TH2F("4D_"+N+"_hResAlphaVsResX", "4D RecHit residual on alpha vs residual on x",
                                 150, -0.3, 0.3, 500, -0.15, 0.15);

      hResAlphaVsResY = new TH2F("4D_"+N+"_hResAlphaVsResY", "4D RecHit residual on alpha vs residual on y",
                                 150, -0.6, 0.6, 500, -0.15, 0.15);

      // Pulls

      hPullAlpha       = new TH1F ("4D_"+N+"_hPullAlpha", 
                                   "4D RecHit pull on #alpha_x direction;(#alpha^{x}_{rec}-#alpha^{x}_{sim})/#sigma",
                                   200, -5, 5);
      hPullAlphaVsEta  = new TH2F ("4D_"+N+"_hPullAlphaVsEta",
                                   "4D RecHit pull on #alpha_x direction vs eta;#eta;(#alpha^{x}_{rec}-#alpha^{x}_{sim})/#sigma",
                                   100, -2.5, 2.5, 100, -5, 5);
      hPullAlphaVsPhi  = new TH2F ("4D_"+N+"_hPullAlphaVsPhi",
                                   "4D RecHit pull on #alpha_x direction vs phi (rad);#phi (rad);(#alpha^{x}_{rec}-#alpha^{x}_{sim})/#sigma",
                                   100, -3.2, 3.2, 100, -5, 5);

      hPullBeta        = new TH1F ("4D_"+N+"_hPullBeta",
                                   "4D RecHit pull on beta direction;(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
                                   200, -5, 5);
      hPullBetaVsEta   = new TH2F ("4D_"+N+"_hPullBetaVsEta",
                                   "4D RecHit pull on beta direction vs eta;#eta;(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
                                   100, -2.5, 2.5, 200, -5, 5);
      hPullBetaVsPhi   = new TH2F ("4D_"+N+"_hPullBetaVsPhi",
                                   "4D RecHit pull on beta direction vs phi;#phi (rad);(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
                                   100, -3.2, 3.2, 200, -5, 5);

      hPullX           = new TH1F ("4D_"+N+"_hPullX",
                                   "4D RecHit pull on position (x) in chamber;(x_{rec}-x_{sim})#sigma",
                                   150, -5, 5);
      hPullXVsEta      = new TH2F ("4D_"+N+"_hPullXVsEta",
                                   "4D RecHit pull on position (x) in chamber vs eta;#eta;(x_{rec}-x_{sim})#sigma",
                                   100, -2.5, 2.5, 150, -5, 5);
      hPullXVsPhi      = new TH2F ("4D_"+N+"_hPullXVsPhi", 
                                   "4D RecHit pull on position (x) in chamber vs phi;#phi (rad);(x_{rec}-x_{sim})/#sigma",
                                   100, -3.2, 3.2, 150, -5, 5);

      hPullY           = new TH1F ("4D_"+N+"_hPullY", 
                                   "4D RecHit pull on position (y) in chamber;(y_{rec}-y_{sim})/#sigma", 150, -5, 5);
      hPullYVsEta      = new TH2F ("4D_"+N+"_hPullYVsEta", 
                                   "4D RecHit pull on position (y) in chamber vs eta;#eta;(y_{rec}-y_{sim})/#sigma",
                                   100, -2.5, 2.5, 150, -5, 5);
      hPullYVsPhi      = new TH2F ("4D_"+N+"_hPullYVsPhi", 
                                   "4D RecHit pull on position (y) in chamber vs phi;#phi (rad);(y_{rec}-y_{sim})/#sigma",
                                   100, -3.2, 3.2, 150, -5, 5);

      // histo in rz SL reference frame.

      hRecBetaRZ        = new TH1F ("4D_"+N+"_hRecBetaRZ", "4D RecHit beta distribution:#alpha^{y} (rad)", 100, -3.5, 3.5);

      hSimBetaRZ      = new TH1F("4D_"+N+"_hSimBetaRZ", "4D segment from SimHit beta distribution in RZ SL;#alpha^{y} (rad)",
                                 100, -3.5, 3.5);
      hRecVsSimBetaRZ = new TH2F("4D_"+N+"_hRecVsSimBetaRZ", "4D segment rec beta vs sim beta (RZ) in RZ SL;#alpha^{y} (rad)",
                                 100, -3.5, 3.5, 100, -3.5, 3.5);

      hResBetaRZ      = new TH1F ("4D_"+N+"_hResBetaRZ",
                                  "4D RecHit residual on beta direction in RZ SL;#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
                                  200, -0.1, 0.1);
      hResBetaVsEtaRZ = new TH2F ("4D_"+N+"_hResBetaVsEtaRZ",
                                  "4D RecHit residual on beta direction vs eta;#eta in RZ SL;#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
                                  100, -2.5, 2.5, 200, -0.2, 0.2);
      hResBetaVsPhiRZ = new TH2F ("4D_"+N+"_hResBetaVsPhiRZ",
                                  "4D RecHit residual on beta direction vs phi in RZ SL;#phi (rad);#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
                                  100, -3.2, 3.2, 200, -0.2, 0.2);

      hResYRZ         = new TH1F ("4D_"+N+"_hResYRZ",
                                  "4D RecHit residual on position (y) in chamber in RZ SL;y_{rec}-y_{sim} (cm)",
                                  150, -0.15, 0.15);
      hResYVsEtaRZ    = new TH2F ("4D_"+N+"_hResYVsEtaRZ",
                                  "4D RecHit residual on position (y) in chamber vs eta in RZ SL;#eta;y_{rec}-y_{sim} (cm)",
                                  100, -2.5, 2.5, 150, -0.6, 0.6);
      hResYVsPhiRZ    = new TH2F ("4D_"+N+"_hResYVsPhiRZ",
                                  "4D RecHit residual on position (y) in chamber vs phi in RZ SL;#phi (rad);y_{rec}-y_{sim} (cm)",
                                  100, -3.2, 3.2, 150, -0.6, 0.6);

      // Pulls
      hPullBetaRZ      = new TH1F ("4D_"+N+"_hPullBetaRZ",
                                   "4D RecHit pull on beta direction in RZ SL;(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
                                   200, -5, 5);
      hPullBetaVsEtaRZ = new TH2F ("4D_"+N+"_hPullBetaVsEtaRZ",
                                   "4D RecHit pull on beta direction vs eta;#eta in RZ SL;(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
                                   100, -2.5, 2.5, 200, -5, 5);
      hPullBetaVsPhiRZ = new TH2F ("4D_"+N+"_hPullBetaVsPhiRZ",
                                   "4D RecHit pull on beta direction vs phi in RZ SL;#phi (rad);(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
                                   100, -3.2, 3.2, 200, -5, 5);

      hPullYRZ         = new TH1F ("4D_"+N+"_hPullYRZ",
                                   "4D RecHit pull on position (y) in chamber in RZ SL;(y_{rec}-y_{sim})/#sigma",
                                   150, -5, 5);
      hPullYVsEtaRZ    = new TH2F ("4D_"+N+"_hPullYVsEtaRZ",
                                   "4D RecHit pull on position (y) in chamber vs eta in RZ SL;#eta;(y_{rec}-y_{sim})/#sigma",
                                   100, -2.5, 2.5, 150, -5, 5);
      hPullYVsPhiRZ    = new TH2F ("4D_"+N+"_hPullYVsPhiRZ",
                                   "4D RecHit pull on position (y) in chamber vs phi in RZ SL;#phi (rad);(y_{rec}-y_{sim})/#sigma",
                                   100, -3.2, 3.2, 150, -5, 5);
    }

    HRes4DHit (TString name_, TFile* file){
      name=name_;

      hRecAlpha = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hRecAlpha");
      hRecBeta = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hRecBeta");

      hSimAlpha = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hSimAlpha");
      hSimBeta = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hSimBeta");

      hRecVsSimAlpha = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hRecVsSimAlpha");
      hRecVsSimBeta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hRecVsSimBeta");

      hResAlpha = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResAlpha");
      hResAlphaVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResAlphaVsEta");
      hResAlphaVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResAlphaVsPhi");

      hResBeta = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResBeta");
      hResBetaVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResBetaVsEta");
      hResBetaVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResBetaVsPhi");

      hResX = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResX");
      hResXVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResXVsEta");
      hResXVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResXVsPhi");

      hResY = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResY");
      hResYVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResYVsEta");
      hResYVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResYVsPhi");

      hResAlphaVsResBeta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResAlphaVsResBeta");
      hResXVsResY = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResXVsResY");
      hResAlphaVsResX = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResAlphaVsResX");
      hResAlphaVsResY = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResAlphaVsResY"); 

      hPullAlpha = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullAlpha");
      hPullAlphaVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullAlphaVsEta");
      hPullAlphaVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullAlphaVsPhi");

      hPullBeta = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullBeta");
      hPullBetaVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullBetaVsEta");
      hPullBetaVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullBetaVsPhi");

      hPullX = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullX");
      hPullXVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullXVsEta");
      hPullXVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullXVsPhi");

      hPullY = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullY");
      hPullYVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullYVsEta");
      hPullYVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullYVsPhi");

      // RX SL frame
      hRecBetaRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hRecBetaRZ");

      hSimBetaRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hSimBetaRZ");

      hRecVsSimBetaRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hRecVsSimBetaRZ");

      hResBetaRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResBetaRZ");
      hResBetaVsEtaRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResBetaVsEtaRZ");
      hResBetaVsPhiRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResBetaVsPhiRZ");

      hResYRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResYRZ");
      hResYVsEtaRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResYVsEtaRZ");
      hResYVsPhiRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hResYVsPhiRZ");

      hPullBetaRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullBetaRZ");
      hPullBetaVsEtaRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullBetaVsEtaRZ");
      hPullBetaVsPhiRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullBetaVsPhiRZ");

      hPullYRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullYRZ");
      hPullYVsEtaRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullYVsEtaRZ");
      hPullYVsPhiRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPullYVsPhiRZ");
    }

    ~HRes4DHit(){
    }

    void Fill(float simDirectionAlpha,
              float recDirectionAlpha,
              float simDirectionBeta,
              float recDirectionBeta,
              float simX,
              float recX,
              float simY,
              float recY,
              float simEta,
              float simPhi,
              float recYRZ,
              float simYRZ,
              float recBetaRZ,
              float simBetaRZ,
              float sigmaAlpha,
              float sigmaBeta,
              float sigmaX,
              float sigmaY,
              float sigmaBetaRZ,
              float sigmaYRZ
             ) {

      hRecAlpha->Fill(recDirectionAlpha);
      hRecBeta->Fill(recDirectionBeta);
      hSimAlpha->Fill(simDirectionAlpha);
      hSimBeta->Fill(simDirectionBeta);

      hRecVsSimAlpha->Fill(simDirectionAlpha, recDirectionAlpha);
      hRecVsSimBeta->Fill(simDirectionBeta, recDirectionBeta);

      float resAlpha = recDirectionAlpha - simDirectionAlpha;
      hResAlpha->Fill(resAlpha);
      hResAlphaVsEta->Fill(simEta, resAlpha);
      hResAlphaVsPhi->Fill(simPhi, resAlpha);
      hPullAlpha->Fill(resAlpha/sigmaAlpha);
      hPullAlphaVsEta->Fill(simEta, resAlpha/sigmaAlpha);
      hPullAlphaVsPhi->Fill(simPhi, resAlpha/sigmaAlpha);
      float resBeta = recDirectionBeta - simDirectionBeta;
      hResBeta->Fill(resBeta);
      hResBetaVsEta->Fill(simEta, resBeta);
      hResBetaVsPhi->Fill(simPhi, resBeta);
      hPullBeta->Fill(resBeta/sigmaBeta);
      hPullBetaVsEta->Fill(simEta, resBeta/sigmaBeta);
      hPullBetaVsPhi->Fill(simPhi, resBeta/sigmaBeta);
      float resX = recX - simX;
      hResX->Fill(resX);
      hResXVsEta->Fill(simEta, resX);
      hResXVsPhi->Fill(simPhi, resX);
      hPullX->Fill(resX/sigmaX);
      hPullXVsEta->Fill(simEta, resX/sigmaX);
      hPullXVsPhi->Fill(simPhi, resX/sigmaX);
      float resY = recY - simY;
      hResY->Fill(resY);
      hResYVsEta->Fill(simEta, resY);
      hResYVsPhi->Fill(simPhi, resY);
      hPullY->Fill(resY/sigmaY);
      hPullYVsEta->Fill(simEta, resY/sigmaY);
      hPullYVsPhi->Fill(simPhi, resY/sigmaY);

      hResAlphaVsResBeta->Fill(resBeta, resAlpha);   
      hResXVsResY->Fill(resY, resX);          
      hResAlphaVsResX->Fill(resX, resAlpha);      
      hResAlphaVsResY->Fill(resY, resAlpha);      

      // RZ SuperLayer
      hRecBetaRZ->Fill(recBetaRZ);
      hSimBetaRZ->Fill(simBetaRZ);

      hRecVsSimBetaRZ->Fill(simBetaRZ, recBetaRZ);

      float resBetaRZ = recBetaRZ - simBetaRZ;
      hResBetaRZ->Fill(resBetaRZ);
      hResBetaVsEtaRZ->Fill(simEta, resBetaRZ);
      hResBetaVsPhiRZ->Fill(simPhi, resBetaRZ);
      hPullBetaRZ->Fill(resBetaRZ/sigmaBetaRZ);
      hPullBetaVsEtaRZ->Fill(simEta, resBetaRZ/sigmaBetaRZ);
      hPullBetaVsPhiRZ->Fill(simPhi, resBetaRZ/sigmaBetaRZ);
      float resYRZ = recYRZ - simYRZ;
      hResYRZ->Fill(resYRZ);
      hResYVsEtaRZ->Fill(simEta, resYRZ);
      hResYVsPhiRZ->Fill(simPhi, resYRZ);
      hPullYRZ->Fill(resYRZ/sigmaYRZ);
      hPullYVsEtaRZ->Fill(simEta, resYRZ/sigmaYRZ);
      hPullYVsPhiRZ->Fill(simPhi, resYRZ/sigmaYRZ);
    }

    void Write() {
      hRecAlpha->Write();
      hRecBeta->Write();
      hSimAlpha->Write();
      hSimBeta->Write();
      hRecVsSimAlpha->Write();
      hRecVsSimBeta->Write();
      hResAlpha->Write();
      hResAlphaVsEta->Write();
      hResAlphaVsPhi->Write();
      hResBeta->Write();
      hResBetaVsEta->Write();
      hResBetaVsPhi->Write();
      hResX->Write();
      hResXVsEta->Write();
      hResXVsPhi->Write();
      hResY->Write();
      hResYVsEta->Write();
      hResYVsPhi->Write();
      hResAlphaVsResBeta->Write();   
      hResXVsResY->Write();
      hResAlphaVsResX->Write();
      hResAlphaVsResY->Write();
      hPullAlpha->Write();
      hPullAlphaVsEta->Write();
      hPullAlphaVsPhi->Write();
      hPullBeta->Write();
      hPullBetaVsEta->Write();
      hPullBetaVsPhi->Write();
      hPullX->Write();
      hPullXVsEta->Write();
      hPullXVsPhi->Write();
      hPullY->Write();
      hPullYVsEta->Write();
      hPullYVsPhi->Write();


      hRecBetaRZ->Write();
      hSimBetaRZ->Write();
      hRecVsSimBetaRZ->Write();
      hResBetaRZ->Write();
      hResBetaVsEtaRZ->Write();
      hResBetaVsPhiRZ->Write();
      hResYRZ->Write();
      hResYVsEtaRZ->Write();
      hResYVsPhiRZ->Write();
      hPullBetaRZ->Write();
      hPullBetaVsEtaRZ->Write();
      hPullBetaVsPhiRZ->Write();
      hPullYRZ->Write();
      hPullYVsEtaRZ->Write();
      hPullYVsPhiRZ->Write();
    }

  public:

    TH1F *hRecAlpha;
    TH1F *hRecBeta;

    TH1F *hSimAlpha;
    TH1F *hSimBeta;

    TH2F *hRecVsSimAlpha;
    TH2F *hRecVsSimBeta;

    TH1F *hResAlpha;
    TH2F *hResAlphaVsEta;
    TH2F *hResAlphaVsPhi;

    TH1F *hResBeta;
    TH2F *hResBetaVsEta;
    TH2F *hResBetaVsPhi;

    TH1F *hResX;
    TH2F *hResXVsEta;
    TH2F *hResXVsPhi;

    TH1F *hResY;
    TH2F *hResYVsEta;
    TH2F *hResYVsPhi;

    TH2F *hResAlphaVsResBeta;   
    TH2F *hResXVsResY;          
    TH2F *hResAlphaVsResX;      
    TH2F *hResAlphaVsResY;      

    TH1F *hPullAlpha;
    TH2F *hPullAlphaVsEta;
    TH2F *hPullAlphaVsPhi;

    TH1F *hPullBeta;
    TH2F *hPullBetaVsEta;
    TH2F *hPullBetaVsPhi;

    TH1F *hPullX;
    TH2F *hPullXVsEta;
    TH2F *hPullXVsPhi;

    TH1F *hPullY;
    TH2F *hPullYVsEta;
    TH2F *hPullYVsPhi;

    // RZ SL 
    TH1F *hRecBetaRZ;

    TH1F *hSimBetaRZ;

    TH2F *hRecVsSimBetaRZ;

    TH1F *hResBetaRZ;
    TH2F *hResBetaVsEtaRZ;
    TH2F *hResBetaVsPhiRZ;

    TH1F *hResYRZ;
    TH2F *hResYVsEtaRZ;
    TH2F *hResYVsPhiRZ;

    TH1F *hPullBetaRZ;
    TH2F *hPullBetaVsEtaRZ;
    TH2F *hPullBetaVsPhiRZ;

    TH1F *hPullYRZ;
    TH2F *hPullYVsEtaRZ;
    TH2F *hPullYVsPhiRZ;

    TString name;
};

//---------------------------------------------------------------------------------------
/// A set of histograms for efficiency 4D RecHits
class HEff4DHit{
  public:
    HEff4DHit(std::string name_){
      TString N = name_.c_str();
      name=N;

      hEtaSimSegm     = new TH1F("4D_"+N+"_hEtaSimSegm", "Eta of SimHit segment", 100, -1.5, 1.5);
      hEtaRecHit      = new TH1F("4D_"+N+"_hEtaRecHit", "Eta distribution of SimHit segment with 4D RecHit",
                                 100, -1.5, 1.5);
      hEffVsEta       = 0;

      hPhiSimSegm     = new TH1F("4D_"+N+"_hPhiSimSegm", "Phi of SimHit segment",
                                 100, -TMath::Pi(),TMath::Pi());
      hPhiRecHit      = new TH1F("4D_"+N+"_hPhiRecHit", "Phi distribution of SimHit segment with 4D RecHit",
                                 100, -TMath::Pi(),TMath::Pi());
      hEffVsPhi       = 0;


      hXSimSegm       = new TH1F("4D_"+N+"_hXSimSegm", "X position in Chamber of SimHit segment (cm)",
                                 100, -200, 200);
      hXRecHit        = new TH1F("4D_"+N+"_hXRecHit", "X position in Chamber of SimHit segment with 4D RecHit (cm)",
                                 100, -200, 200);
      hEffVsX         = 0;

      hYSimSegm       = new TH1F("4D_"+N+"_hYSimSegm", "Y position in Chamber of SimHit segment (cm)",
                                 100, -200, 200);
      hYRecHit        = new TH1F("4D_"+N+"_hYRecHit", "Y position in Chamber of SimHit segment with 4D RecHit (cm)",
                                 100, -200, 200);
      hEffVsY         = 0;

      hAlphaSimSegm   = new TH1F("4D_"+N+"_hAlphaSimSegm", "Alpha of SimHit segment (rad)",
                                 100, -1.5, 1.5);
      hAlphaRecHit    = new TH1F("4D_"+N+"_hAlphaRecHit", "Alpha of SimHit segment with 4D RecHit (rad)",
                                 100, -1.5, 1.5);
      hEffVsAlpha     = 0;

      hBetaSimSegm   = new TH1F("4D_"+N+"_hBetaSimSegm", "Beta of SimHit segment (rad)",
                                100, -2, 2);
      hBetaRecHit    = new TH1F("4D_"+N+"_hBetaRecHit", "Beta of SimHit segment with 4D RecHit (rad)",
                                100, -2, 2);
      hEffVsBeta     = 0;

    }

    HEff4DHit (TString name_, TFile* file){
      name=name_;
      hEtaSimSegm = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hEtaSimSegm");
      hEtaRecHit = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hEtaRecHit");
      hEffVsEta = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hEffVsEta");

      hPhiSimSegm = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPhiSimSegm");
      hPhiRecHit = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hPhiRecHit");
      hEffVsPhi = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hEffVsPhi");

      hXSimSegm  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hXSimSegm");
      hXRecHit  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hXRecHit");
      hEffVsX  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hEffVsX");

      hYSimSegm  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hYSimSegm");
      hYRecHit  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hYRecHit");
      hEffVsY  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hEffVsY");

      hAlphaSimSegm  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hAlphaSimSegm");
      hAlphaRecHit  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hAlphaRecHit");
      hEffVsAlpha  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hEffVsAlpha");

      hBetaSimSegm  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hBetaSimSegm");
      hBetaRecHit  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hBetaRecHit");
      hEffVsBeta  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hEffVsBeta");
    }


    ~HEff4DHit(){

      /*delete hEtaSimSegm;   
        delete hEtaRecHit;    
        delete hEffVsEta;     
        delete hPhiSimSegm;   
        delete hPhiRecHit;    
        delete hEffVsPhi;     
        delete hXSimSegm;     
        delete hXRecHit;      
        delete hEffVsX;       
        delete hYSimSegm;     
        delete hYRecHit;      
        delete hEffVsY;       
        delete hAlphaSimSegm; 
        delete hAlphaRecHit;  
        delete hEffVsAlpha;   
        delete hBetaSimSegm;  
        delete hBetaRecHit;   
        delete hEffVsBeta;*/    
    }

    void Fill(float etaSimSegm,
              float phiSimSegm,
              float xSimSegm,
              float ySimSegm,
              float alphaSimSegm,
              float betaSimSegm,
              bool fillRecHit) {

      hEtaSimSegm->Fill(etaSimSegm);
      hPhiSimSegm->Fill(phiSimSegm);
      hXSimSegm->Fill(xSimSegm);
      hYSimSegm->Fill(ySimSegm);
      hAlphaSimSegm->Fill(alphaSimSegm);
      hBetaSimSegm->Fill(betaSimSegm);

      if(fillRecHit) {
        hEtaRecHit->Fill(etaSimSegm);  
        hPhiRecHit->Fill(phiSimSegm);
        hXRecHit->Fill(xSimSegm);
        hYRecHit->Fill(ySimSegm);
        hAlphaRecHit->Fill(alphaSimSegm);
        hBetaRecHit->Fill(betaSimSegm);
      }
    }



    void ComputeEfficiency() {

      hEffVsEta = (TH1F *) hEtaRecHit->Clone();
      hEffVsEta->SetName("4D_"+name+"_hEffVsEta");
      hEffVsEta->SetTitle("4D RecHit Efficiency as a function of Eta");
      hEffVsEta->Divide(hEtaSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsEta = hEffVsEta->GetNbinsX();
      for(int bin = 1; bin <=  nBinsEta; bin++) {
        float nSimHit = hEtaSimSegm->GetBinContent(bin);
        float eff = hEffVsEta->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsEta->SetBinError(bin, error);
      }

      hEffVsPhi = (TH1F *) hPhiRecHit->Clone();
      hEffVsPhi->SetName("4D_"+name+"_hEffVsPhi");
      hEffVsPhi->SetTitle("4D RecHit Efficiency as a function of Phi");
      hEffVsPhi->Divide(hPhiSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsPhi = hEffVsPhi->GetNbinsX();
      for(int bin = 1; bin <=  nBinsPhi; bin++) {
        float nSimHit = hPhiSimSegm->GetBinContent(bin);
        float eff = hEffVsPhi->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsPhi->SetBinError(bin, error);
      }

      hEffVsX = (TH1F *) hXRecHit->Clone();
      hEffVsX->SetName("4D_"+name+"_hEffVsX");
      hEffVsX->SetTitle("4D RecHit Efficiency as a function of x position in Chamber");
      hEffVsX->Divide(hXSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsX = hEffVsX->GetNbinsX();
      for(int bin = 1; bin <=  nBinsX; bin++) {
        float nSimHit = hXSimSegm->GetBinContent(bin);
        float eff = hEffVsX->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsX->SetBinError(bin, error);
      }


      hEffVsY = (TH1F *) hYRecHit->Clone();
      hEffVsY->SetName("4D_"+name+"_hEffVsY");
      hEffVsY->SetTitle("4D RecHit Efficiency as a function of y position in Chamber");
      hEffVsY->Divide(hYSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsY = hEffVsY->GetNbinsX();
      for(int bin = 1; bin <=  nBinsY; bin++) {
        float nSimHit = hYSimSegm->GetBinContent(bin);
        float eff = hEffVsY->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsY->SetBinError(bin, error);
      }

      hEffVsAlpha = (TH1F *) hAlphaRecHit->Clone();
      hEffVsAlpha->SetName("4D_"+name+"_hEffVsAlpha");
      hEffVsAlpha->SetTitle("4D RecHit Efficiency as a function of alpha");
      hEffVsAlpha->Divide(hAlphaSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsAlpha = hEffVsAlpha->GetNbinsX();
      for(int bin = 1; bin <=  nBinsAlpha; bin++) {
        float nSimHit = hAlphaSimSegm->GetBinContent(bin);
        float eff = hEffVsAlpha->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsAlpha->SetBinError(bin, error);
      }


      hEffVsBeta = (TH1F *) hBetaRecHit->Clone();
      hEffVsBeta->SetName("4D_"+name+"_hEffVsBeta");
      hEffVsBeta->SetTitle("4D RecHit Efficiency as a function of beta");
      hEffVsBeta->Divide(hBetaSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsBeta = hEffVsBeta->GetNbinsX();
      for(int bin = 1; bin <=  nBinsBeta; bin++) {
        float nSimHit = hBetaSimSegm->GetBinContent(bin);
        float eff = hEffVsBeta->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsBeta->SetBinError(bin, error);
      }
    }

    void Write() {
      hEtaSimSegm->Write();   
      hEtaRecHit->Write();
      if(hEffVsEta != 0)
        hEffVsEta->Write();     
      hPhiSimSegm->Write();   
      hPhiRecHit->Write();    
      if(hEffVsPhi != 0)
        hEffVsPhi->Write();     
      hXSimSegm->Write();     
      hXRecHit->Write();      
      if(hEffVsX != 0)
        hEffVsX->Write();       
      hYSimSegm->Write();     
      hYRecHit->Write();      
      if(hEffVsY != 0)
        hEffVsY->Write();       
      hAlphaSimSegm->Write(); 
      hAlphaRecHit->Write();  
      if(hEffVsAlpha != 0)
        hEffVsAlpha->Write();   
      hBetaSimSegm->Write();  
      hBetaRecHit->Write();   
      if(hEffVsBeta != 0)
        hEffVsBeta->Write();    

    }

  public:
    TH1F *hEtaSimSegm;   
    TH1F *hEtaRecHit;    
    TH1F *hEffVsEta;     
    TH1F *hPhiSimSegm;   
    TH1F *hPhiRecHit;    
    TH1F *hEffVsPhi;     
    TH1F *hXSimSegm;     
    TH1F *hXRecHit;      
    TH1F *hEffVsX;       
    TH1F *hYSimSegm;     
    TH1F *hYRecHit;      
    TH1F *hEffVsY;       
    TH1F *hAlphaSimSegm; 
    TH1F *hAlphaRecHit;  
    TH1F *hEffVsAlpha;   
    TH1F *hBetaSimSegm;  
    TH1F *hBetaRecHit;   
    TH1F *hEffVsBeta;    

    TString name;

};


#endif

