#ifndef RecoLocalMuon_Histograms_H
#define RecoLocalMuon_Histograms_H


/** \class Histograms
 *  Collection of histograms for DT RecHit and Segment test.
 *
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


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
    HRes1DHit(std::string name_,DQMStore* dbe_,bool doall=true,bool local=true){
      std::string pre ="1D_";
      pre += name_;
      _doall = doall;
      // Position, sigma, residual, pull
      //sprintf (histo_n, "1D_%s_hDist",N); sprintf(histo_t, "1D RHit distance from wire");
      //      hDist=0; hDist = dbe_->book1D(histo_n, histo_t, 100, 0,2.5);
      dbe_->setCurrentFolder("DT/1DRecHits/Res/");
      
      if(doall){
	hDist=0; hDist = dbe_->book1D(pre + "_hDist" ,"1D RHit distance from wire", 100, 0,2.5);
	//hDist       = new TH1F ("1D_"+N+"_hDist", "1D RHit distance from wire", 100, 0,2.5);
	hResVsAngle = 0; hResVsAngle   = dbe_->book2D(pre+"_hResVsAngle", "1D RHit residual vs impact angle",100, -1.2,1.2, 100, -0.2,0.2);    
	hResVsDistFE = 0; hResVsDistFE = dbe_->book2D(pre+"_hResVsDistFE", "1D RHit residual vs FE distance", 100, 0.,400., 150, -0.5,0.5);    
	dbe_->setCurrentFolder("DT/1DRecHits/Pull/");
 	hPullVsPos= 0; hPullVsPos  = dbe_->book2D (pre+"_hPullVsPos", "1D RHit pull vs position", 100, 0,2.5, 100, -5,5);
	hPullVsAngle = 0; hPullVsAngle  = dbe_->book2D (pre+"_hPullVsAngle", "1D RHit pull vs impact angle",100, -1.2,1.2, 100, -5,5);
	hPullVsDistFE = 0; hPullVsDistFE  = dbe_->book2D (pre+"_hPullVsDistFE", "1D RHit pull vs FE distance", 100, 0., 400., 100, -5,5);
      }
      dbe_->setCurrentFolder("DT/1DRecHits/Res/");
      hRes=0; hRes = dbe_->book1D(pre + "_hRes","1D RHit residual", 300, -0.5,0.5);
      hResSt[0] = 0; hResSt[0] = dbe_->book1D(pre + "_hResMB1","1D RHit residual", 300, -0.5,0.5);
      hResSt[1] = 0; hResSt[1] = dbe_->book1D(pre + "_hResMB2","1D RHit residual", 300, -0.5,0.5);
      hResSt[2] = 0; hResSt[2] = dbe_->book1D(pre + "_hResMB3","1D RHit residual", 300, -0.5,0.5);
      hResSt[3] = 0; hResSt[3] = dbe_->book1D(pre + "_hResMB4","1D RHit residual", 300, -0.5,0.5);

      //hRes        = new TH1F ("1D_"+N+"_hRes", "1D RHit residual", 300, -1.5,1.5);
      hResVsEta=0; hResVsEta = dbe_->book2D(pre +"_hResVsEta" , "1D RHit residual vs eta", 50, -1.25,1.25,150,-0.5,0.5);
      //hResVsEta   = new TH2F("1D_"+N+"_hResVsEta", "1D RHit residual vs eta",50, -1.25,1.25, 150, -1.5,1.5);
      hResVsPhi = 0; hResVsPhi   = dbe_->book2D(pre+"_hResVsPhi" , "1D RHit residual vs phi",100, -3.2, 3.2, 150, -0.5,0.5);
      //hResVsPhi   = new TH2F("1D_"+N+"_hResVsPhi", "1D RHit residual vs phi", 100, -3.2, 3.2, 150, -1.5,1.5);
      hResVsPos = 0; hResVsPos   = dbe_->book2D(pre+"_hResVsPos", "1D RHit residual vs position",100, 0, 2.5, 150, -0.5,0.5);    
      //hResVsPos   = new TH2F("1D_"+N+"_hResVsPos", "1D RHit residual vs position",100, 0, 2.5, 150, -1.5,1.5);    
      dbe_->setCurrentFolder("DT/1DRecHits/Pull/");
      hPull =0; hPull       = dbe_->book1D (pre+"_hPull", "1D RHit pull", 100, -5,5);
      hPullSt[0] = 0; hPullSt[0] = dbe_->book1D(pre + "_hPullMB1","1D RHit residual", 100, -5,5);
      hPullSt[1] = 0; hPullSt[1] = dbe_->book1D(pre + "_hPullMB2","1D RHit residual", 100, -5,5);
      hPullSt[2] = 0; hPullSt[2] = dbe_->book1D(pre + "_hPullMB3","1D RHit residual", 100, -5,5);
      hPullSt[3] = 0; hPullSt[3] = dbe_->book1D(pre + "_hPullMB4","1D RHit residual", 100, -5,5);
    }
  
//     HRes1DHit(TString name_, TFile* file){
//       name=name_;
//       base = "/DQMData/Run 1/DT/Run summary/1DRecHits/Res/";
//       hDist          = (TH1F *) file->Get(base+"1D_"+name+"_hDist");
//       hRes           = (TH1F *) file->Get(base+"1D_"+name+"_hRes");
//       hResVsEta      = (TH2F *) file->Get(base+"1D_"+name+"_hResVsEta");
//       hResVsPhi      = (TH2F *) file->Get(base+"1D_"+name+"_hResVsPhi");
//       hResVsPos      = (TH2F *) file->Get(base+"1D_"+name+"_hResVsPos");
//       hResVsAngle    = (TH2F *) file->Get(base+"1D_"+name+"_hResVsAngle");
//       hResVsDistFE   = (TH2F *) file->Get(base+"1D_"+name+"_hResVsDistFE");
//       hPull          = (TH1F *) file->Get(base+"1D_"+name+"_hPull");
//       hPullVsPos     = (TH2F *) file->Get(base+"1D_"+name+"_hPullVsPos");
//       hPullVsAngle   = (TH2F *) file->Get(base+"1D_"+name+"_hPullVsAngle");
//       hPullVsDistFE  = (TH2F *) file->Get(base+"1D_"+name+"_hPullVsDistFE");
//     }


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
              float errRecHit,
	      int station) {
      // Reso, pull
      float res = distRecHit-distSimHit;
      if(_doall){
	hDist->Fill(distRecHit);
	hResVsAngle->Fill(thetaSimHit,res);
	hResVsDistFE->Fill(distFESimHit,res);
      }
      hRes->Fill(res); hResSt[station-1]->Fill(res);
      hResVsEta->Fill(etaSimHit,res);
      hResVsPhi->Fill(phiSimHit,res);
      hResVsPos->Fill(distSimHit,res);
      if(errRecHit!=0) {
        float pull=res/errRecHit;
        hPull->Fill(pull);hPullSt[station-1]->Fill(pull);
	if(_doall){
	  hPullVsPos->Fill(distSimHit,pull);
	  hPullVsAngle->Fill(thetaSimHit,pull);
	  hPullVsDistFE->Fill(distFESimHit,pull);
	}
      }
      else std::cout<<"Error: RecHit error = 0" << std::endl;
    }

    /*void Write() {
      hDist->Write();     
      hRes->Write();      
      hResVsEta->Write();   
      hResVsPhi->Write(); 
      hResVsPos->Write(); 
      hResVsAngle->Write(); 
      hResVsDistFE->Write(); 
      hPull->Write();
      hPullVsPos->Write();
      hPullVsAngle->Write();
      hPullVsDistFE->Write();
      }*/


  public:
    MonitorElement* hDist;
    MonitorElement* hRes;
    MonitorElement* hResSt[4];
    MonitorElement* hResVsEta;
    MonitorElement* hResVsPhi;
    MonitorElement* hResVsPos;
    MonitorElement* hResVsAngle;
    MonitorElement* hResVsDistFE;

    MonitorElement* hPull;
    MonitorElement* hPullSt[4];
    MonitorElement* hPullVsPos;
    MonitorElement* hPullVsAngle;
    MonitorElement* hPullVsDistFE;
    bool _doall;
    TString name;

};


//---------------------------------------------------------------------------------------
class HEff1DHit{
  public:
    HEff1DHit(std::string name_,DQMStore *dbe_){
      std::string pre ="1D_";
      pre += name_;
      name = pre;
      dbe_->setCurrentFolder("DT/1DRecHits/");
      hEtaMuSimHit=0; hEtaMuSimHit = dbe_->book1D(pre+"_hEtaMuSimHit", "SimHit Eta distribution",100, -1.5, 1.5);
      hEtaRecHit=0; hEtaRecHit = dbe_->book1D(pre+"_hEtaRecHit", "SimHit Eta distribution with 1D RecHit",100, -1.5, 1.5);
      hEffVsEta = 0;
      hPhiMuSimHit=0; hPhiMuSimHit = dbe_->book1D(pre+"_hPhiMuSimHit", "SimHit Phi distribution",100, -TMath::Pi(),TMath::Pi());
      hPhiRecHit=0; hPhiRecHit = dbe_->book1D(pre+"_hPhiRecHit", "SimHit Phi distribution with 1D RecHit",100, -TMath::Pi(),TMath::Pi());
      hEffVsPhi = 0;
      hDistMuSimHit=0;hDistMuSimHit = dbe_->book1D(pre+"_hDistMuSimHit", "SimHit Distance from wire distribution",100, 0, 2.5);
      hDistRecHit=0; hDistRecHit = dbe_->book1D(pre+"_hDistRecHit", "SimHit Distance from wire distribution with 1D RecHit",100, 0, 2.5);
      hEffVsDist = 0;

    }
/*
    HEff1DHit (TString name_, TFile* file){
      name=name_;
      hEtaMuSimHit        = (TH1F *) file->Get("/DQMData/1D_"+name+"_hEtaMuSimHit");
      hEtaRecHit          = (TH1F *) file->Get("/DQMData/1D_"+name+"_hEtaRecHit");
      hEffVsEta           = (TH1F *) file->Get("/DQMData/1D_"+name+"_hEffVsEta");

      hPhiMuSimHit        = (TH1F *) file->Get("/DQMData/1D_"+name+"_hPhiMuSimHit");
      hPhiRecHit          = (TH1F *) file->Get("/DQMData/1D_"+name+"_hPhiRecHit");
      hEffVsPhi           = (TH1F *) file->Get("/DQMData/1D_"+name+"_hEffVsPhi");

      hDistMuSimHit       = (TH1F *) file->Get("/DQMData/1D_"+name+"_hDistMuSimHit");
      hDistRecHit         = (TH1F *) file->Get("/DQMData/1D_"+name+"_hDistRecHit");
      hEffVsDist          = (TH1F *) file->Get("/DQMData/1D_"+name+"_hEffVsDist");
      }
*/

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


      //      hEffVsEta = (TH1F *) hEtaRecHit->Clone();
      hEffVsEta = (TH1F *) hEtaRecHit->getTH1();
      TH1F * hEffEtaMuSimHit = (TH1F *) hEtaMuSimHit->getTH1();
      hEffVsEta->SetTitle("1D RecHit Efficiency as a function of Eta");
      hEffVsEta->SetName(name+"_hEffVsEta");
      //hEffVsEta->Divide(hEtaMuSimHit);
      hEffVsEta->Divide(hEffEtaMuSimHit);
      // Set the error accordingly to binomial statistics
      int nBinsEta = hEffVsEta->GetNbinsX();
      for(int bin = 1; bin <=  nBinsEta; bin++) {
        float nSimHit = hEffEtaMuSimHit->GetBinContent(bin);
        float eff = hEffVsEta->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsEta->SetBinError(bin, error);
      }

      //hEffVsPhi = (TH1F *) hPhiRecHit->Clone();
      hEffVsPhi = (TH1F *) hPhiRecHit->getTH1();
      TH1F * hEffPhiMuSimHit = (TH1F *) hPhiMuSimHit->getTH1();
      hEffVsPhi->SetTitle("1D RecHit Efficiency as a function of Phi");
      hEffVsPhi->SetName(name+"_hEffVsPhi");
      hEffVsPhi->Divide(hEffPhiMuSimHit);
      // Set the error accordingly to binomial statistics
      int nBinsPhi = hEffVsPhi->GetNbinsX();
      for(int bin = 1; bin <=  nBinsPhi; bin++) {
        float nSimHit = hEffPhiMuSimHit->GetBinContent(bin);
        float eff = hEffVsPhi->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsPhi->SetBinError(bin, error);
      }

      //hEffVsDist = (TH1F *) hDistRecHit->Clone();
      hEffVsDist = (TH1F *) hDistRecHit->getTH1();
      TH1F * hEffDistMuSimHit = (TH1F *) hDistMuSimHit->getTH1();
      hEffVsDist->SetTitle("1D RecHit Efficiency as a function of Dist");
      hEffVsDist->SetName(name+"_hEffVsDist");
      hEffVsDist->Divide(hEffDistMuSimHit);
      // Set the error accordingly to binomial statistics
      int nBinsDist = hEffVsDist->GetNbinsX();
      for(int bin = 1; bin <=  nBinsDist; bin++) {
        float nSimHit = hEffDistMuSimHit->GetBinContent(bin);
        float eff = hEffVsDist->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsDist->SetBinError(bin, error);
      }


    }

    /*void Write() {
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
	}*/

  public:
    MonitorElement* hEtaMuSimHit;
    MonitorElement* hEtaRecHit;
    TH1F* hEffVsEta;

    MonitorElement* hPhiMuSimHit;
    MonitorElement* hPhiRecHit;
    TH1F* hEffVsPhi;

    MonitorElement* hDistMuSimHit;
    MonitorElement* hDistRecHit;
    TH1F* hEffVsDist;

    TString name;

};

//---------------------------------------------------------------//

// Histos of residuals for 2D rechits
class HRes2DHit{
  public:
    HRes2DHit(std::string name_,DQMStore* dbe_,bool doall=true,bool local=true){
      _doall = doall;
      std::string pre ="2D_";
      pre += name_;
      dbe_->setCurrentFolder("DT/2DSegments/Res/");
      if(doall){
	hRecAngle=0;hRecAngle = dbe_->book1D (pre+"_hRecAngle", "Distribution of Rec segment angles;angle (rad)",100, -1.5, 1.5);
	hSimAngle=0;hSimAngle = dbe_->book1D (pre+"_hSimAngle", "Distribution of segment angles from SimHits;angle (rad)",100, -1.5, 1.5);
	hRecVsSimAngle=0;hRecVsSimAngle = dbe_->book2D (pre+"_hRecVsSimAngle", "Rec angle vs sim angle;angle (rad)",100, -1.5, 1.5, 100, -1.5, 1.5);
	hResAngleVsEta=0;hResAngleVsEta   = dbe_->book2D (pre+"_hResAngleVsEta", "Residual on 2D segment angle vs Eta; #eta; res (rad)",100, -2.5, 2.5, 200, -0.2, 0.2);
	hResAngleVsPhi=0;hResAngleVsPhi   = dbe_->book2D (pre+"_hResAngleVsPhi", "Residual on 2D segment angle vs Phi; #phi (rad);res (rad)",
							  100, -3.2, 3.2, 150, -0.2, 0.2);
	hResPosVsEta=0;hResPosVsEta   = dbe_->book2D (pre+"_hResPosVsEta", "Residual on 2D segment position vs Eta;#eta;res (cm)",
						      100, -2.5, 2.5, 150, -0.2, 0.2);
	hResPosVsPhi=0;hResPosVsPhi   = dbe_->book2D (pre+"_hResPosVsPhi", "Residual on 2D segment position vs Phi;#phi (rad);res (cm)",
						      100, -3.2, 3.2, 150, -0.2, 0.2);
	
	hResPosVsResAngle=0;hResPosVsResAngle = dbe_->book2D(pre+"_hResPosVsResAngle",
							     "Residual on 2D segment position vs Residual on 2D segment angle;angle (rad);res (cm)",
							     100, -0.3, 0.3, 150, -0.2, 0.2);
	
      }
      
      hResAngle=0; hResAngle   = dbe_->book1D (pre+"_hResAngle", "Residual on 2D segment angle;angle_{rec}-angle_{sim} (rad)", 50, -0.01, 0.01);
      
      hResPos=0;hResPos   = dbe_->book1D (pre+"_hResPos", "Residual on 2D segment position (x at SL center);x_{rec}-x_{sim} (cm)",
					  150, -0.2, 0.2);
      dbe_->setCurrentFolder("DT/2DSegments/Pull/");
     
      hPullAngle=0;hPullAngle   = dbe_->book1D (pre+"_hPullAngle", "Pull on 2D segment angle;(angle_{rec}-angle_{sim})/#sigma (rad)", 150, -5, 5);
      hPullPos=0;hPullPos   = dbe_->book1D (pre+"_hPullPos", "Pull on 2D segment position (x at SL center);(x_{rec}-x_{sim} (cm))/#sigma",
					    150, -5, 5);
    }

/*    HRes2DHit (TString name_, TFile* file){
      name=name_;

      hRecAngle = (TH1F *) file->Get("DQMData/2D_"+name+"_hRecAngle");
      hSimAngle = (TH1F *) file->Get("DQMData/2D_"+name+"_hSimAngle");
      hRecVsSimAngle = (TH2F *) file->Get("DQMData/2D_"+name+"_hRecVsSimAngle");
      hResAngle = (TH1F *) file->Get("DQMData/2D_"+name+"_hResAngle");
      hResAngleVsEta = (TH2F *) file->Get("DQMData/2D_"+name+"_hResAngleVsEta");
      hResAngleVsPhi = (TH2F *) file->Get("DQMData/2D_"+name+"_hResAngleVsPhi");
      hResPos = (TH1F *) file->Get("DQMData/2D_"+name+"_hResPos");
      hResPosVsEta = (TH2F *) file->Get("DQMData/2D_"+name+"_hResPosVsEta");
      hResPosVsPhi = (TH2F *) file->Get("DQMData/2D_"+name+"_hResPosVsPhi");
      hResPosVsResAngle = (TH2F *) file->Get("DQMData/2D_"+name+"_hResPosVsResAngle");
      hPullAngle = (TH1F *) file->Get("DQMData/2D_"+name+"_hPullAngle");
      hPullPos = (TH1F *) file->Get("DQMData/2D_"+name+"_hPullPos");

    }
   */

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

      float resAngle = angleRecSegment-angleSimSegment;
      hResAngle->Fill(resAngle);
      float resPos = posRecSegment-posSimSegment;
      hResPos->Fill(resPos);
      hPullAngle->Fill(resAngle/sigmaAngle);
      hPullPos->Fill(resPos/sigmaPos);
      if(_doall){
	hRecAngle->Fill(angleRecSegment);
	hSimAngle->Fill(angleSimSegment);
	hRecVsSimAngle->Fill(angleSimSegment, angleRecSegment);
	hResAngleVsEta->Fill(etaSimSegment, resAngle);
	hResAngleVsPhi->Fill(phiSimSegment, resAngle);
	hResPosVsEta->Fill(etaSimSegment, resPos);
	hResPosVsPhi->Fill(phiSimSegment, resPos);
	hResPosVsResAngle->Fill(resAngle, resPos);
      }
    }

    /*void Write() {

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
      }*/


  public:
    MonitorElement *hRecAngle;
    MonitorElement *hSimAngle;
    MonitorElement *hRecVsSimAngle;
    MonitorElement *hResAngle;
    MonitorElement *hResAngleVsEta;
   MonitorElement  *hResAngleVsPhi;
    MonitorElement *hResPos;
    MonitorElement *hResPosVsEta;
   MonitorElement  *hResPosVsPhi;
    MonitorElement *hResPosVsResAngle;
    MonitorElement *hPullAngle;
    MonitorElement *hPullPos;
    TString name;
    bool _doall; 

};

//--------------------------------------------------------------------------------//

// Histos for 2D RecHit efficiency
class HEff2DHit{
  public:
    HEff2DHit(std::string name_, DQMStore * dbe_){
      std::string pre ="2D_";
      pre += name_;
      name = pre;
      dbe_->setCurrentFolder("DT/2DSegments/");
      hEtaSimSegm=0;hEtaSimSegm     = dbe_->book1D(pre+"_hEtaSimSegm", "Eta of SimHit segment", 100, -1.5, 1.5);
      hEtaRecHit=0;hEtaRecHit      = dbe_->book1D(pre+"_hEtaRecHit", "Eta distribution of SimHit segment with 2D RecHit",
                                 100, -1.5, 1.5);
      //hEffVsEta       = 0;

      hPhiSimSegm=0;hPhiSimSegm     = dbe_->book1D(pre+"_hPhiSimSegm", "Phi of SimHit segment",
                                 100, -TMath::Pi(),TMath::Pi());
      hPhiRecHit=0;hPhiRecHit      = dbe_->book1D(pre+"_hPhiRecHit", "Phi distribution of SimHit segment with 2D RecHit",
                                 100, -TMath::Pi(),TMath::Pi());
      //hEffVsPhi       = 0;


      hPosSimSegm=0;hPosSimSegm     = dbe_->book1D(pre+"_hPosSimSegm", "Position in SL of SimHit segment (cm)",
                                 100, -250, 250);
      hPosRecHit=0;hPosRecHit      = dbe_->book1D(pre+"_hPosRecHit", "Position in SL of SimHit segment with 2D RecHit (cm)",
                                 100, -250, 250);
      //hEffVsPos       = 0;


      hAngleSimSegm=0;hAngleSimSegm   = dbe_->book1D(pre+"_hAngleSimSegm", "Angle of SimHit segment (rad)",
                                 100, -2, 2);
      hAngleRecHit=0;hAngleRecHit    = dbe_->book1D(pre+"_hAngleRecHit", "Angle of SimHit segment with 2D RecHit (rad)",
                                 100, -2, 2);
      hEffVsAngle=0;hEffVsAngle     = 0;

    }
/*
    HEff2DHit (TString name_, TFile* file){
      name=name_;
      hEtaSimSegm = (TH1F *) file->Get("DQMData/2D_"+name+"_hEtaSimSegm");
      hEtaRecHit = (TH1F *) file->Get("DQMData/2D_"+name+"_hEtaRecHit");
      hEffVsEta = (TH1F *) file->Get("DQMData/2D_"+name+"_hEffVsEta");

      hPhiSimSegm = (TH1F *) file->Get("DQMData/2D_"+name+"_hPhiSimSegm");
      hPhiRecHit = (TH1F *) file->Get("DQMData/2D_"+name+"_hPhiRecHit");
      hEffVsPhi = (TH1F *) file->Get("DQMData/2D_"+name+"_hEffVsPhi");

      hPosSimSegm = (TH1F *) file->Get("DQMData/2D_"+name+"_hPosSimSegm");
      hPosRecHit = (TH1F *) file->Get("DQMData/2D_"+name+"_hPosRecHit");
      hEffVsPos = (TH1F *) file->Get("DQMData/2D_"+name+"_hEffVsPos");

      hAngleSimSegm = (TH1F *) file->Get("DQMData/2D_"+name+"_hAngleSimSegm");
      hAngleRecHit = (TH1F *) file->Get("DQMData/2D_"+name+"_hAngleRecHit");
      hEffVsAngle = (TH1F *) file->Get("DQMData/2D_"+name+"_hEffVsAngle");
      }
*/

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

      //hEffVsEta = (TH1F *) hEtaRecHit->Clone();
      hEffVsEta = (TH1F *) hEtaRecHit->getTH1();
      TH1F * hEffEtaSimSegm = (TH1F *) hEtaSimSegm->getTH1();
      hEffVsEta->SetTitle("2D RecHit Efficiency as a function of Eta");
      hEffVsEta->SetName(name+"_hEffVsEta");
      hEffVsEta->Divide(hEffEtaSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsEta = hEffVsEta->GetNbinsX();
      for(int bin = 1; bin <=  nBinsEta; bin++) {
        float nSimHit = hEffEtaSimSegm->GetBinContent(bin);
        float eff = hEffVsEta->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsEta->SetBinError(bin, error);
      }

      //hEffVsPhi = (TH1F *) hPhiRecHit->Clone();
      hEffVsPhi = (TH1F *) hPhiRecHit->getTH1();
      TH1F * hEffPhiSimSegm = (TH1F *) hPhiSimSegm->getTH1();
      hEffVsPhi->SetTitle("2D RecHit Efficiency as a function of Phi");
      hEffVsPhi->SetName(name+"_hEffVsPhi");
      hEffVsPhi->Divide(hEffPhiSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsPhi = hEffVsPhi->GetNbinsX();
      for(int bin = 1; bin <=  nBinsPhi; bin++) {
        float nSimHit = hEffPhiSimSegm->GetBinContent(bin);
        float eff = hEffVsPhi->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsPhi->SetBinError(bin, error);
      }

      hEffVsPos = (TH1F *) hPosRecHit->getTH1();
      TH1F * hEffPosSimSegm = (TH1F *) hPosSimSegm->getTH1();
      //hEffVsPos = (TH1F *) hPosRecHit->Clone();
      hEffVsPos->SetName(name+"_hEffVsPos");
      hEffVsPos->SetTitle("2D RecHit Efficiency as a function of position in SL");
      hEffVsPos->Divide(hEffPosSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsPos = hEffVsPos->GetNbinsX();
      for(int bin = 1; bin <=  nBinsPos; bin++) {
        float nSimHit = hEffPosSimSegm->GetBinContent(bin);
        float eff = hEffVsPos->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsPos->SetBinError(bin, error);
      }

      //hEffVsAngle = (TH1F *) hAngleRecHit->Clone();
      hEffVsAngle = (TH1F *) hAngleRecHit->getTH1();
      TH1F * hEffAngleSimSegm = (TH1F *) hAngleSimSegm->getTH1();
      hEffVsAngle->SetTitle("2D RecHit Efficiency as a function of angle");
      hEffVsAngle->SetName(name+"_hEffVsAngle");
      hEffVsAngle->Divide(hEffAngleSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsAngle = hEffVsAngle->GetNbinsX();
      for(int bin = 1; bin <=  nBinsAngle; bin++) {
        float nSimHit = hEffAngleSimSegm->GetBinContent(bin);
        float eff = hEffVsAngle->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsAngle->SetBinError(bin, error);
      }

    }

    /*void Write() {
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

	}*/

  public:

    MonitorElement *hEtaSimSegm;   
    MonitorElement  *hEtaRecHit;    
    TH1F  *hEffVsEta;     
    MonitorElement  *hPhiSimSegm;   
    MonitorElement  *hPhiRecHit;    
    TH1F  *hEffVsPhi;     
    MonitorElement  *hPosSimSegm;   
    MonitorElement  *hPosRecHit;    
    TH1F  *hEffVsPos;     
    MonitorElement  *hAngleSimSegm; 
    MonitorElement  *hAngleRecHit;  
    TH1F   *hEffVsAngle;   
    
    TString name;

};

//---------------------------------------------------------------------------------------
// Histos of residuals for 4D rechits
class HRes4DHit{
  public:
  HRes4DHit(std::string name_,DQMStore *dbe_,bool doall=true,bool local=true) : isLocal(local){
      std::string pre ="4D_";
      pre += name_;
      _doall = doall;
      
      dbe_->setCurrentFolder("DT/4DSegments/Res/");    
      if(doall){
	hRecAlpha=0;hRecAlpha  = dbe_->book1D (pre+"_hRecAlpha", "4D RecHit alpha (RPhi) distribution;#alpha^{x} (rad)", 100, -1.5, 1.5);
	hRecBeta=0;hRecBeta = dbe_->book1D (pre+"_hRecBeta", "4D RecHit beta distribution:#alpha^{y} (rad)", 100, -1.5, 1.5);
	
	hSimAlpha=0;hSimAlpha = dbe_->book1D(pre+"_hSimAlpha", "4D segment from SimHit alpha (RPhi) distribution;i#alpha^{x} (rad)",
					     100, -1.5, 1.5);
	hSimBeta=0;hSimBeta = dbe_->book1D(pre+"_hSimBeta", "4D segment from SimHit beta distribution;#alpha^{y} (rad)",
					   100, -1.5, 1.5);
	hRecVsSimAlpha=0;hRecVsSimAlpha = dbe_->book2D(pre+"_hRecVsSimAlpha", "4D segment rec alpha {v}s sim alpha (RPhi);#alpha^{x} (rad)",
						       100, -1.5, 1.5, 100, -1.5, 1.5);
	hRecVsSimBeta=0;hRecVsSimBeta = dbe_->book2D(pre+"_hRecVsSimBeta", "4D segment rec beta vs sim beta (RZ);#alpha^{y} (rad)",
						     100, -1.5, 1.5, 100, -1.5, 1.5);
	
	hResAlphaVsEta=0;hResAlphaVsEta = dbe_->book2D (pre+"_hResAlphaVsEta",
							"4D RecHit residual on #alpha_x direction vs eta;#eta;#alpha^{x}_{rec}-#alpha^{x}_{sim} (rad)",
							100, -2.5, 2.5, 100, -0.025, 0.025);
	hResAlphaVsPhi=0;hResAlphaVsPhi = dbe_->book2D (pre+"_hResAlphaVsPhi",
							"4D RecHit residual on #alpha_x direction vs phi (rad);#phi (rad);#alpha^{x}_{rec}-#alpha^{x}_{sim} (rad)",
							100, -3.2, 3.2, 100, -0.025, 0.025);
	hResBetaVsEta=0;hResBetaVsEta = dbe_->book2D (pre+"_hResBetaVsEta",
						      "4D RecHit residual on beta direction vs eta;#eta;#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
						      100, -2.5, 2.5, 200, -0.2, 0.2);
	hResBetaVsPhi=0;hResBetaVsPhi = dbe_->book2D (pre+"_hResBetaVsPhi",
						      "4D RecHit residual on beta direction vs phi;#phi (rad);#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
						      100, -3.2, 3.2, 200, -0.2, 0.2);
	
	hResXVsEta=0;hResXVsEta = dbe_->book2D (pre+"_hResXVsEta", "4D RecHit residual on position (x) in chamber vs eta;#eta;x_{rec}-x_{sim} (cm)",
						100, -2.5, 2.5, 150, -0.3, 0.3);
	hResXVsPhi=0;hResXVsPhi= dbe_->book2D (pre+"_hResXVsPhi", "4D RecHit residual on position (x) in chamber vs phi;#phi (rad);x_{rec}-x_{sim} (cm)",
					       100, -3.2, 3.2, 150, -0.3, 0.3);
	
	hResYVsEta=0;hResYVsEta = dbe_->book2D (pre+"_hResYVsEta", "4D RecHit residual on position (y) in chamber vs eta;#eta;y_{rec}-y_{sim} (cm)",
						100, -2.5, 2.5, 150, -0.6, 0.6);
	hResYVsPhi=0;hResYVsPhi = dbe_->book2D (pre+"_hResYVsPhi", "4D RecHit residual on position (y) in chamber vs phi;#phi (rad);y_{rec}-y_{sim} (cm)",
						100, -3.2, 3.2, 150, -0.6, 0.6);
	
	hResAlphaVsResBeta=0;hResAlphaVsResBeta = dbe_->book2D(pre+"_hResAlphaVsResBeta", "4D RecHit residual on alpha vs residual on beta",
							       200, -0.3, 0.3, 500, -0.15, 0.15);
	hResXVsResY=0;hResXVsResY= dbe_->book2D(pre+"_hResXVsResY", "4D RecHit residual on X vs residual on Y",
						150, -0.6, 0.6, 50, -0.3, 0.3);
	hResAlphaVsResX=0;hResAlphaVsResX = dbe_->book2D(pre+"_hResAlphaVsResX", "4D RecHit residual on alpha vs residual on x",
							 150, -0.3, 0.3, 500, -0.15, 0.15);
	
	hResAlphaVsResY=0;hResAlphaVsResY = dbe_->book2D(pre+"_hResAlphaVsResY", "4D RecHit residual on alpha vs residual on y",
							 150, -0.6, 0.6, 500, -0.15, 0.15);
	
	hRecBetaRZ=0;hRecBetaRZ        = dbe_->book1D (pre+"_hRecBetaRZ", "4D RecHit beta distribution:#alpha^{y} (rad)", 100, -1.5, 1.5);
	
	hSimBetaRZ=0;hSimBetaRZ      = dbe_->book1D(pre+"_hSimBetaRZ", "4D segment from SimHit beta distribution in RZ SL;#alpha^{y} (rad)",
						    100, -1.5, 1.5);
	hRecVsSimBetaRZ=0;hRecVsSimBetaRZ = dbe_->book2D(pre+"_hRecVsSimBetaRZ", "4D segment rec beta vs sim beta (RZ) in RZ SL;#alpha^{y} (rad)",
							 100, -1.5, 1.5, 100, -1.5, 1.5);
	
	hResBetaVsEtaRZ=0;hResBetaVsEtaRZ = dbe_->book2D (pre+"_hResBetaVsEtaRZ",
							  "4D RecHit residual on beta direction vs eta;#eta in RZ SL;#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
							  100, -2.5, 2.5, 200, -0.2, 0.2);
	hResBetaVsPhiRZ=0;hResBetaVsPhiRZ = dbe_->book2D (pre+"_hResBetaVsPhiRZ",
							  "4D RecHit residual on beta direction vs phi in RZ SL;#phi (rad);#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
							  100, -3.2, 3.2, 200, -0.2, 0.2);
	hResYVsEtaRZ=0;hResYVsEtaRZ    = dbe_->book2D (pre+"_hResYVsEtaRZ",
						       "4D RecHit residual on position (y) in chamber vs eta in RZ SL;#eta;y_{rec}-y_{sim} (cm)",
						       100, -2.5, 2.5, 150, -0.6, 0.6);
	hResYVsPhiRZ=0;hResYVsPhiRZ    = dbe_->book2D (pre+"_hResYVsPhiRZ",
						       "4D RecHit residual on position (y) in chamber vs phi in RZ SL;#phi (rad);y_{rec}-y_{sim} (cm)",
						       100, -3.2, 3.2, 150, -0.6, 0.6);
	dbe_->setCurrentFolder("DT/4DSegments/Pull/");    
 	hPullAlphaVsEta=0;hPullAlphaVsEta  = dbe_->book2D (pre+"_hPullAlphaVsEta",
							   "4D RecHit pull on #alpha_x direction vs eta;#eta;(#alpha^{x}_{rec}-#alpha^{x}_{sim})/#sigma",
							   100, -2.5, 2.5, 100, -5, 5);
	hPullAlphaVsPhi=0;hPullAlphaVsPhi  = dbe_->book2D (pre+"_hPullAlphaVsPhi",
							   "4D RecHit pull on #alpha_x direction vs phi (rad);#phi (rad);(#alpha^{x}_{rec}-#alpha^{x}_{sim})/#sigma",
							   100, -3.2, 3.2, 100, -5, 5);
	hPullBetaVsEta=0;hPullBetaVsEta   = dbe_->book2D (pre+"_hPullBetaVsEta",
							  "4D RecHit pull on beta direction vs eta;#eta;(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
							  100, -2.5, 2.5, 200, -5, 5);
	hPullBetaVsPhi=0;hPullBetaVsPhi   = dbe_->book2D (pre+"_hPullBetaVsPhi",
							  "4D RecHit pull on beta direction vs phi;#phi (rad);(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
							  100, -3.2, 3.2, 200, -5, 5);
	hPullXVsEta=0;hPullXVsEta      = dbe_->book2D (pre+"_hPullXVsEta",
						       "4D RecHit pull on position (x) in chamber vs eta;#eta;(x_{rec}-x_{sim})#sigma",
						       100, -2.5, 2.5, 150, -5, 5);
	hPullXVsPhi=0;hPullXVsPhi      = dbe_->book2D (pre+"_hPullXVsPhi", 
						       "4D RecHit pull on position (x) in chamber vs phi;#phi (rad);(x_{rec}-x_{sim})/#sigma",
						       100, -3.2, 3.2, 150, -5, 5);
	hPullYVsEta=0;hPullYVsEta      = dbe_->book2D (pre+"_hPullYVsEta", 
						       "4D RecHit pull on position (y) in chamber vs eta;#eta;(y_{rec}-y_{sim})/#sigma",
						       100, -2.5, 2.5, 150, -5, 5);
	hPullYVsPhi=0;hPullYVsPhi      = dbe_->book2D (pre+"_hPullYVsPhi", 
						       "4D RecHit pull on position (y) in chamber vs phi;#phi (rad);(y_{rec}-y_{sim})/#sigma",
						       100, -3.2, 3.2, 150, -5, 5);
	hPullBetaVsEtaRZ=0;hPullBetaVsEtaRZ = dbe_->book2D (pre+"_hPullBetaVsEtaRZ",
                                   "4D RecHit pull on beta direction vs eta;#eta in RZ SL;(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
							    100, -2.5, 2.5, 200, -5, 5);
	hPullBetaVsPhiRZ=0;hPullBetaVsPhiRZ = dbe_->book2D (pre+"_hPullBetaVsPhiRZ",
							    "4D RecHit pull on beta direction vs phi in RZ SL;#phi (rad);(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
							    100, -3.2, 3.2, 200, -5, 5);
	hPullYVsEtaRZ=0;hPullYVsEtaRZ    = dbe_->book2D (pre+"_hPullYVsEtaRZ",
							 "4D RecHit pull on position (y) in chamber vs eta in RZ SL;#eta;(y_{rec}-y_{sim})/#sigma",
							 100, -2.5, 2.5, 150, -5, 5);
	hPullYVsPhiRZ=0;hPullYVsPhiRZ    = dbe_->book2D (pre+"_hPullYVsPhiRZ",
                                   "4D RecHit pull on position (y) in chamber vs phi in RZ SL;#phi (rad);(y_{rec}-y_{sim})/#sigma",
                                   100, -3.2, 3.2, 150, -5, 5);

      }
      dbe_->setCurrentFolder("DT/4DSegments/Res/");    
      hResAlpha=0;hResAlpha = dbe_->book1D (pre+"_hResAlpha", 
                                  "4D RecHit residual on #alpha_x direction;#alpha^{x}_{rec}-#alpha^{x}_{sim} (rad)",
					    200, -0.015, 0.015);

      hResBeta=0;hResBeta = dbe_->book1D (pre+"_hResBeta",
                                  "4D RecHit residual on beta direction;#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
                                  200, -0.1, 0.1);
      hResX=0;hResX = dbe_->book1D (pre+"_hResX", "4D RecHit residual on position (x) in chamber;x_{rec}-x_{sim} (cm)",
                                  150, -0.15, 0.15);
      hResY=0;hResY = dbe_->book1D (pre+"_hResY", "4D RecHit residual on position (y) in chamber;y_{rec}-y_{sim} (cm)", 150, -0.6, 0.6);

      // histo in rz SL reference frame.

      hResBetaRZ=0;hResBetaRZ      = dbe_->book1D (pre+"_hResBetaRZ",
                                  "4D RecHit residual on beta direction in RZ SL;#alpha^{y}_{rec}-#alpha^{y}_{sim} (rad)",
                                  200, -0.1, 0.1);

      hResYRZ=0;hResYRZ         = dbe_->book1D (pre+"_hResYRZ",
                                  "4D RecHit residual on position (y) in chamber in RZ SL;y_{rec}-y_{sim} (cm)",
                                  150, -0.15, 0.15);

      // Pulls
      dbe_->setCurrentFolder("DT/4DSegments/Pull/");    

      hPullAlpha=0;hPullAlpha = dbe_->book1D (pre+"_hPullAlpha", 
                                   "4D RecHit pull on #alpha_x direction;(#alpha^{x}_{rec}-#alpha^{x}_{sim})/#sigma",
                                   200, -5, 5);
      hPullBeta=0;hPullBeta  = dbe_->book1D (pre+"_hPullBeta",
                                   "4D RecHit pull on beta direction;(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
                                   200, -5, 5);

      hPullX=0;hPullX           = dbe_->book1D (pre+"_hPullX",
                                   "4D RecHit pull on position (x) in chamber;(x_{rec}-x_{sim})#sigma",
                                   150, -5, 5);

      hPullY=0;hPullY          = dbe_->book1D (pre+"_hPullY", 
                                   "4D RecHit pull on position (y) in chamber;(y_{rec}-y_{sim})/#sigma", 150, -5, 5);

     hPullBetaRZ=0;hPullBetaRZ      = dbe_->book1D (pre+"_hPullBetaRZ",
                                   "4D RecHit pull on beta direction in RZ SL;(#alpha^{y}_{rec}-#alpha^{y}_{sim})/#sigma",
                                   200, -5, 5);

      hPullYRZ=0;hPullYRZ         = dbe_->book1D (pre+"_hPullYRZ",
                                   "4D RecHit pull on position (y) in chamber in RZ SL;(y_{rec}-y_{sim})/#sigma",
                                   150, -5, 5);

      // NHits, t0
      if (isLocal) {
	dbe_->setCurrentFolder("DT/4DSegments/");
	hHitMult                  = dbe_->book2D(pre+"_hNHits", "NHits", 12,0,12, 6,0,6);
	ht0                       = dbe_->book2D(pre+"_ht0",    "t0",    200,-25,25,200,-25,25);
      }

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
              float sigmaYRZ,
	      int nHitsPhi,
	      int nHitsTheta,
	      float t0Phi,
	      float t0Theta
             ) {
      float resAlpha = recDirectionAlpha - simDirectionAlpha;
      hResAlpha->Fill(resAlpha);
      hPullAlpha->Fill(resAlpha/sigmaAlpha);
      float resBeta = recDirectionBeta - simDirectionBeta;
      hResBeta->Fill(resBeta);
      hPullBeta->Fill(resBeta/sigmaBeta);
      float resX = recX - simX;
      hResX->Fill(resX);
      hPullX->Fill(resX/sigmaX);
      float resY = recY - simY;
      hResY->Fill(resY);
      hPullY->Fill(resY/sigmaY);


      float resBetaRZ = recBetaRZ - simBetaRZ;
      hResBetaRZ->Fill(resBetaRZ);
      hPullBetaRZ->Fill(resBetaRZ/sigmaBetaRZ);
      float resYRZ = recYRZ - simYRZ;
      hResYRZ->Fill(resYRZ);
      hPullYRZ->Fill(resYRZ/sigmaYRZ);
      if(_doall){
	hRecAlpha->Fill(recDirectionAlpha);
	hRecBeta->Fill(recDirectionBeta);
	hSimAlpha->Fill(simDirectionAlpha);
	hSimBeta->Fill(simDirectionBeta);
	hRecVsSimAlpha->Fill(simDirectionAlpha, recDirectionAlpha);
	hRecVsSimBeta->Fill(simDirectionBeta, recDirectionBeta);
	hResAlphaVsEta->Fill(simEta, resAlpha);
	hResAlphaVsPhi->Fill(simPhi, resAlpha);
	hPullAlphaVsEta->Fill(simEta, resAlpha/sigmaAlpha);
	hPullAlphaVsPhi->Fill(simPhi, resAlpha/sigmaAlpha);
	hResBetaVsEta->Fill(simEta, resBeta);
	hResBetaVsPhi->Fill(simPhi, resBeta);
	hPullBetaVsEta->Fill(simEta, resBeta/sigmaBeta);
	hPullBetaVsPhi->Fill(simPhi, resBeta/sigmaBeta);
	hResXVsEta->Fill(simEta, resX);
	hResXVsPhi->Fill(simPhi, resX);
	hPullXVsEta->Fill(simEta, resX/sigmaX);
	hPullXVsPhi->Fill(simPhi, resX/sigmaX);
	hResYVsEta->Fill(simEta, resY);
	hResYVsPhi->Fill(simPhi, resY);
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
	hResBetaVsEtaRZ->Fill(simEta, resBetaRZ);
	hResBetaVsPhiRZ->Fill(simPhi, resBetaRZ);
	hPullBetaVsEtaRZ->Fill(simEta, resBetaRZ/sigmaBetaRZ);
	hPullBetaVsPhiRZ->Fill(simPhi, resBetaRZ/sigmaBetaRZ);
	hResYVsEtaRZ->Fill(simEta, resYRZ);
	hResYVsPhiRZ->Fill(simPhi, resYRZ);
	hPullYVsEtaRZ->Fill(simEta, resYRZ/sigmaYRZ);
	hPullYVsPhiRZ->Fill(simPhi, resYRZ/sigmaYRZ);
      }
      if (isLocal){
	hHitMult->Fill(nHitsPhi, nHitsTheta);
	ht0->Fill(t0Phi,t0Theta);
      }
    }

  public:

    MonitorElement *hRecAlpha;
    MonitorElement *hRecBeta;

    MonitorElement *hSimAlpha;
    MonitorElement *hSimBeta;

    MonitorElement *hRecVsSimAlpha;
    MonitorElement *hRecVsSimBeta;

    MonitorElement *hResAlpha;
    MonitorElement *hResAlphaVsEta;
    MonitorElement *hResAlphaVsPhi;

    MonitorElement *hResBeta;
    MonitorElement *hResBetaVsEta;
    MonitorElement *hResBetaVsPhi;

    MonitorElement *hResX;
    MonitorElement *hResXVsEta;
    MonitorElement  *hResXVsPhi;

    MonitorElement *hResY;
    MonitorElement *hResYVsEta;
    MonitorElement *hResYVsPhi;

    MonitorElement *hResAlphaVsResBeta;   
    MonitorElement *hResXVsResY;          
    MonitorElement *hResAlphaVsResX;      
    MonitorElement *hResAlphaVsResY;      

    MonitorElement *hPullAlpha;
    MonitorElement *hPullAlphaVsEta;
    MonitorElement *hPullAlphaVsPhi;

    MonitorElement *hPullBeta;
    MonitorElement *hPullBetaVsEta;
    MonitorElement *hPullBetaVsPhi;

    MonitorElement *hPullX;
    MonitorElement *hPullXVsEta;
    MonitorElement *hPullXVsPhi;

    MonitorElement *hPullY;
    MonitorElement *hPullYVsEta;
    MonitorElement *hPullYVsPhi;

    // RZ SL 
    MonitorElement *hRecBetaRZ;

    MonitorElement *hSimBetaRZ;

    MonitorElement *hRecVsSimBetaRZ;

    MonitorElement *hResBetaRZ;
    MonitorElement *hResBetaVsEtaRZ;
    MonitorElement *hResBetaVsPhiRZ;

    MonitorElement *hResYRZ;
    MonitorElement *hResYVsEtaRZ;
    MonitorElement *hResYVsPhiRZ;

    MonitorElement *hPullBetaRZ;
    MonitorElement *hPullBetaVsEtaRZ;
    MonitorElement *hPullBetaVsPhiRZ;

    MonitorElement *hPullYRZ;
    MonitorElement *hPullYVsEtaRZ;
    MonitorElement *hPullYVsPhiRZ;

    MonitorElement *hHitMult;
    MonitorElement *ht0;

    bool _doall;
    bool isLocal;
    TString name;    
};

//---------------------------------------------------------------------------------------
/// A set of histograms for efficiency 4D RecHits
class HEff4DHit{
  public:
    HEff4DHit(std::string name_,DQMStore *dbe_){
      std::string pre ="4D_";
      pre += name_;
      name = pre;
      dbe_->setCurrentFolder("DT/4DSegments/");     
      hEtaSimSegm=0;hEtaSimSegm     = dbe_->book1D(pre+"_hEtaSimSegm", "Eta of SimHit segment", 100, -1.5, 1.5);
      hEtaRecHit=0;hEtaRecHit      = dbe_->book1D(pre+"_hEtaRecHit", "Eta distribution of SimHit segment with 4D RecHit",
                                 100, -1.5, 1.5);
      hEffVsEta       = 0;

      hPhiSimSegm=0;hPhiSimSegm     = dbe_->book1D(pre+"_hPhiSimSegm", "Phi of SimHit segment",
                                 100, -TMath::Pi(),TMath::Pi());
      hPhiRecHit=0;hPhiRecHit      = dbe_->book1D(pre+"_hPhiRecHit", "Phi distribution of SimHit segment with 4D RecHit",
                                 100, -TMath::Pi(),TMath::Pi());
      hEffVsPhi       = 0;


      hXSimSegm=0;hXSimSegm       = dbe_->book1D(pre+"_hXSimSegm", "X position in Chamber of SimHit segment (cm)",
                                 100, -200, 200);
      hXRecHit=0;hXRecHit        = dbe_->book1D(pre+"_hXRecHit", "X position in Chamber of SimHit segment with 4D RecHit (cm)",
                                 100, -200, 200);
      hEffVsX         = 0;

      hYSimSegm=0;hYSimSegm       = dbe_->book1D(pre+"_hYSimSegm", "Y position in Chamber of SimHit segment (cm)",
                                 100, -200, 200);
      hYRecHit=0;hYRecHit        = dbe_->book1D(pre+"_hYRecHit", "Y position in Chamber of SimHit segment with 4D RecHit (cm)",
                                 100, -200, 200);
      hEffVsY         = 0;

      hAlphaSimSegm=0;hAlphaSimSegm   = dbe_->book1D(pre+"_hAlphaSimSegm", "Alpha of SimHit segment (rad)",
                                 100, -1.5, 1.5);
      hAlphaRecHit=0;hAlphaRecHit    = dbe_->book1D(pre+"_hAlphaRecHit", "Alpha of SimHit segment with 4D RecHit (rad)",
                                 100, -1.5, 1.5);
      hEffVsAlpha     = 0;

      hBetaSimSegm=0;hBetaSimSegm   = dbe_->book1D(pre+"_hBetaSimSegm", "Beta of SimHit segment (rad)",
                                100, -2, 2);
      hBetaRecHit=0;hBetaRecHit    = dbe_->book1D(pre+"_hBetaRecHit", "Beta of SimHit segment with 4D RecHit (rad)",
                                100, -2, 2);
      hEffVsBeta     = 0;

      hNSeg =0; hNSeg            = dbe_->book1D(pre+"_hNSeg", "Number of rec segment per sim seg",
                                 20, 0, 20);

    }

    ~HEff4DHit(){

    }
    
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
      //hEffVsEta = (TH1F *) hEtaRecHit->Clone();
      hEffVsEta = (TH1F *) hEtaRecHit->getTH1();
      TH1F * hEffEtaSimSegm = (TH1F *) hEtaSimSegm->getTH1();
      hEffVsEta->SetName(name+"_hEffVsEta");
      hEffVsEta->SetTitle("4D RecHit Efficiency as a function of Eta");
      hEffVsEta->Divide(hEffEtaSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsEta = hEffVsEta->GetNbinsX();
      for(int bin = 1; bin <=  nBinsEta; bin++) {
        float nSimHit = hEffEtaSimSegm->GetBinContent(bin);
        float eff = hEffVsEta->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsEta->SetBinError(bin, error);
      }

      hEffVsPhi = (TH1F *) hPhiRecHit->getTH1();
      TH1F * hEffPhiSimSegm = (TH1F *) hPhiSimSegm->getTH1();
      hEffVsPhi->SetName(name+"_hEffVsPhi");
      hEffVsPhi->SetTitle("4D RecHit Efficiency as a function of Phi");
      hEffVsPhi->Divide(hEffPhiSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsPhi = hEffVsPhi->GetNbinsX();
      for(int bin = 1; bin <=  nBinsPhi; bin++) {
        float nSimHit = hEffPhiSimSegm->GetBinContent(bin);
        float eff = hEffVsPhi->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsPhi->SetBinError(bin, error);
      }

      hEffVsX = (TH1F *) hXRecHit->getTH1();
      TH1F * hEffXSimSegm = (TH1F *) hXSimSegm->getTH1();
      hEffVsX->SetName(name+"_hEffVsX");
      hEffVsX->SetTitle("4D RecHit Efficiency as a function of x position in Chamber");
      hEffVsX->Divide(hEffXSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsX = hEffVsX->GetNbinsX();
      for(int bin = 1; bin <=  nBinsX; bin++) {
        float nSimHit = hEffXSimSegm->GetBinContent(bin);
        float eff = hEffVsX->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsX->SetBinError(bin, error);
      }


      hEffVsY = (TH1F *) hYRecHit->getTH1();
      TH1F * hEffYSimSegm = (TH1F *) hYSimSegm->getTH1();
      hEffVsY->SetName(name+"_hEffVsY");
      hEffVsY->SetTitle("4D RecHit Efficiency as a function of y position in Chamber");
      hEffVsY->Divide(hEffYSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsY = hEffVsY->GetNbinsX();
      for(int bin = 1; bin <=  nBinsY; bin++) {
        float nSimHit = hEffYSimSegm->GetBinContent(bin);
        float eff = hEffVsY->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsY->SetBinError(bin, error);
      }

      hEffVsAlpha = (TH1F *) hAlphaRecHit->getTH1();
      TH1F * hEffAlphaSimSegm = (TH1F *) hAlphaSimSegm->getTH1();
      hEffVsAlpha->SetName(name+"_hEffVsAlpha");
      hEffVsAlpha->SetTitle("4D RecHit Efficiency as a function of alpha");
      hEffVsAlpha->Divide(hEffAlphaSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsAlpha = hEffVsAlpha->GetNbinsX();
      for(int bin = 1; bin <=  nBinsAlpha; bin++) {
        float nSimHit = hEffAlphaSimSegm->GetBinContent(bin);
        float eff = hEffVsAlpha->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsAlpha->SetBinError(bin, error);
      }


      hEffVsBeta = (TH1F *) hBetaRecHit->getTH1();
      TH1F * hEffBetaSimSegm = (TH1F *) hBetaSimSegm->getTH1();
      hEffVsBeta->SetName(name+"_hEffVsBeta");
      hEffVsBeta->SetTitle("4D RecHit Efficiency as a function of beta");
      hEffVsBeta->Divide(hEffBetaSimSegm);
      // Set the error accordingly to binomial statistics
      int nBinsBeta = hEffVsBeta->GetNbinsX();
      for(int bin = 1; bin <=  nBinsBeta; bin++) {
        float nSimHit = hEffBetaSimSegm->GetBinContent(bin);
        float eff = hEffVsBeta->GetBinContent(bin);
        float error = 0;
        if(nSimHit != 0) {
          error = sqrt(eff*(1-eff)/nSimHit);
        }
        hEffVsBeta->SetBinError(bin, error);
      }
    }

  public:
    MonitorElement *hEtaSimSegm;   
    MonitorElement *hEtaRecHit;    
    TH1F *hEffVsEta;     
    MonitorElement *hPhiSimSegm;   
    MonitorElement *hPhiRecHit;    
    TH1F *hEffVsPhi;     
    MonitorElement *hXSimSegm;     
    MonitorElement *hXRecHit;      
    TH1F *hEffVsX;       
    MonitorElement *hYSimSegm;     
    MonitorElement *hYRecHit;      
    TH1F *hEffVsY;       
    MonitorElement *hAlphaSimSegm; 
    MonitorElement *hAlphaRecHit;  
    TH1F *hEffVsAlpha;   
    MonitorElement *hBetaSimSegm;  
    MonitorElement *hBetaRecHit;   
    TH1F *hEffVsBeta;    
    MonitorElement *hNSeg;    

    TString name;

};
			       

#endif

