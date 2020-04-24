#ifndef RecoLocalMuon_Histograms_H
#define RecoLocalMuon_Histograms_H

/** \class Histograms
 *  Collection of histograms for DT RecHit and Segment test.
 *  This interface is intended only for reading histogram sets from root files. cf ../plugins/Histograms.h
 *
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


// Retrieve standard name string for histogram sets. Note that validation plots are currently done per abs(wheel).
TString buildName(int wheel, int station, int sl) {
  TString name_;
  if(sl == 0) {
    name_+="W";    
  } else if(sl == 2) {
    name_+="RZ_W";
  } else {
    name_+="RPhi_W";
  }
  if (station==0) {
    name_+=long(abs(wheel));
  } else {
    name_=name_+long(abs(wheel))+"_St"+long(station);
  }
  return name_;
}
  



//---------------------------------------------------------------------------------------
/// A set of histograms of residuals and pulls for 1D RecHits
class HRes1DHit{
  public:

  HRes1DHit(TFile* file, int wheel, int station, int sl, const TString& step){
    TString name_=step;
    name_+=buildName(wheel,station,sl);
    initFromFile(name_,file);
  }
  

  HRes1DHit(TString name_, TFile* file){
    initFromFile(name_,file);
  }
  

  void initFromFile(TString name_, TFile* file){
    name=name_;
    hDist          = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Res/1D_"+name+"_hDist");
    hRes           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Res/1D_"+name+"_hRes");
    hResSt[0]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Res/1D_"+name+"_hResMB1");
    hResSt[1]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Res/1D_"+name+"_hResMB2");
    hResSt[2]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Res/1D_"+name+"_hResMB3");
    hResSt[3]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Res/1D_"+name+"_hResMB4");
    hResVsEta      = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Res/1D_"+name+"_hResVsEta");
    hResVsPhi      = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Res/1D_"+name+"_hResVsPhi");
    hResVsPos      = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Res/1D_"+name+"_hResVsPos");
    hResVsAngle    = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Res/1D_"+name+"_hResVsAngle");
    hResVsDistFE   = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Res/1D_"+name+"_hResVsDistFE");
    hPull          = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Pull/1D_"+name+"_hPull");
    hPullSt[0]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Pull/1D_"+name+"_hPullMB1");
    hPullSt[1]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Pull/1D_"+name+"_hPullMB2");
    hPullSt[2]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Pull/1D_"+name+"_hPullMB3");
    hPullSt[3]           = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Pull/1D_"+name+"_hPullMB4");
    hPullVsPos     = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Pull/1D_"+name+"_hPullVsPos");
    hPullVsAngle   = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Pull/1D_"+name+"_hPullVsAngle");
    hPullVsDistFE  = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/1DRecHits/Pull/1D_"+name+"_hPullVsDistFE");

    if (hRes) {
      hRes->SetXTitle("|d_{hit}|-|d_{true}| (cm)");
      hResVsPos->SetXTitle("|d_{true}| (cm)");
      hResVsPos->SetYTitle("|d_{hit}|-|d_{true}| (cm)");
      hResVsAngle->SetXTitle("#alpha_{true} (rad)");
      hResVsAngle->SetYTitle("|d_{hit}|-|d_{true}| (cm)");
    }
  }


    ~HRes1DHit(){}

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
  
    HEff1DHit (TFile* file, int wheel, int station, int sl, const TString& step){
      TString name_=step;
      name_+=buildName(wheel,station,sl);
      initFromFile(name_,file);
    }
  

    HEff1DHit (TString name_, TFile* file){
      initFromFile(name_,file);
    }
  
    void initFromFile (TString name_, TFile* file){
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

      ComputeEfficiency();

    }


    ~HEff1DHit(){}

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

      if (hEffVsEta!=0 || hEffVsPhi!=0 || hEffVsDist!=0) {
	cout << "ComputeEfficiency: histogram already present" << endl;
	abort();
      }
      
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


    ~HRes2DHit(){}

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


    ~HEff2DHit(){}

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
  
    HRes4DHit (TFile* file, int wheel, int station, int sl){
      initFromFile(buildName(wheel,station,sl),file);
    }
  

    HRes4DHit (TString name_, TFile* file){
      initFromFile(name_,file);
    }
  

    void initFromFile (TString name_, TFile* file){
      name=name_;

      hRecAlpha = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hRecAlpha");
      hRecBeta = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hRecBeta");

      hSimAlpha = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hSimAlpha");
      hSimBeta = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hSimBeta");

      hRecVsSimAlpha = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hRecVsSimAlpha");
      hRecVsSimBeta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hRecVsSimBeta");

      hResAlpha = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResAlpha");
      hResAlphaVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResAlphaVsEta");
      hResAlphaVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResAlphaVsPhi");

      hResBeta = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResBeta");
      hResBetaVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResBetaVsEta");
      hResBetaVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResBetaVsPhi");

      hResX = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResX");
      hResXVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResXVsEta");
      hResXVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResXVsPhi");

      hResY = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResY");
      hResYVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResYVsEta");
      hResYVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResYVsPhi");

      hResAlphaVsResBeta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResAlphaVsResBeta");
      hResXVsResY = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResXVsResY");
      hResAlphaVsResX = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResAlphaVsResX");
      hResAlphaVsResY = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResAlphaVsResY"); 

      hPullAlpha = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullAlpha");
      hPullAlphaVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullAlphaVsEta");
      hPullAlphaVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullAlphaVsPhi");

      hPullBeta = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullBeta");
      hPullBetaVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullBetaVsEta");
      hPullBetaVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullBetaVsPhi");

      hPullX = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullX");
      hPullXVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullXVsEta");
      hPullXVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullXVsPhi");

      hPullY = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullY");
      hPullYVsEta = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullYVsEta");
      hPullYVsPhi = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullYVsPhi");

      // RX SL frame
      hRecBetaRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hRecBetaRZ");

      hSimBetaRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hSimBetaRZ");

      hRecVsSimBetaRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hRecVsSimBetaRZ");

      hResBetaRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResBetaRZ");
      hResBetaVsEtaRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResBetaVsEtaRZ");
      hResBetaVsPhiRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResBetaVsPhiRZ");

      hResYRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResYRZ");
      hResYVsEtaRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResYVsEtaRZ");
      hResYVsPhiRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Res/4D_"+name+"_hResYVsPhiRZ");

      hPullBetaRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullBetaRZ");
      hPullBetaVsEtaRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullBetaVsEtaRZ");
      hPullBetaVsPhiRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullBetaVsPhiRZ");

      hPullYRZ = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullYRZ");
      hPullYVsEtaRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullYVsEtaRZ");
      hPullYVsPhiRZ = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/Pull/4D_"+name+"_hPullYVsPhiRZ");

      hHitMult = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hNHits");
      ht0      = (TH2F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_ht0");
	
    }

    ~HRes4DHit(){}

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

    TH2F* hHitMult;
    TH2F *ht0;

    TString name;
};

//---------------------------------------------------------------------------------------
/// A set of histograms for efficiency 4D RecHits
class HEff4DHit{
  
 public: 
  
  HEff4DHit (TFile* file, int wheel, int station, int sl){
    initFromFile(buildName(wheel,station,sl),file);
  }
  
  
  HEff4DHit (TString name_, TFile* file){
    initFromFile(name_,file);
  }
  
  
  
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

    hNSeg          = new TH1F(" 4D_"+N+"_hNSeg", "Number of rec segment per sim seg",
                                 20, 0, 20);

  }
  
  void initFromFile (TString name_, TFile* file){
  
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

      hNSeg  = (TH1F *) file->Get("DQMData/Run 1/DT/Run summary/4DSegments/4D_"+name+"_hNSeg");

      ComputeEfficiency();

    }


    ~HEff4DHit(){}

    void Fill(float etaSimSegm,
              float phiSimSegm,
              float xSimSegm,
              float ySimSegm,
              float alphaSimSegm,
              float betaSimSegm,
              bool fillRecHit,
              int nSeg) {

      hEtaSimSegm->Fill(etaSimSegm);
      hPhiSimSegm->Fill(phiSimSegm);
      hXSimSegm->Fill(xSimSegm);
      hYSimSegm->Fill(ySimSegm);
      hAlphaSimSegm->Fill(alphaSimSegm);
      hBetaSimSegm->Fill(betaSimSegm);
      hNSeg->Fill(nSeg);

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
    TH1F *hNSeg;    
 
    TString name;

};


#endif

