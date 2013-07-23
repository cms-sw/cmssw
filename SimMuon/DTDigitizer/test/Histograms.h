#ifndef HISTOGRAMS_H
#define HISTOGRAMS_H

/** \class Histograms
 *  Classes for histograms handling.
 *
 *  $Date: 2007/09/05 07:29:06 $
 *  $Revision: 1.5 $
 *  \author R. Bellan - INFN Torino
 */
#include "TString.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TFile.h"

#include <string>
#include <iostream>

#include "DataFormats/GeometryVector/interface/Pi.h"

class hDigis{
 public:
  hDigis(std::string name_){
    TString N = name_.c_str();
    name=N;
    //booking degli istogrammi unidimensionali
    hMuonDigis = new TH1F(N+"_hMuonDigis", "number of muon digis", 20, 0., 20.);
    hMuonTimeDigis  = new TH1F(N+"_hMuonTimeDigis", "Muon digis time box", 2048, 0., 1600.);
    //    control = new TH1F (N+"_control", "control", 2048, 0., 1600.);
    //2D
    hMuonTimeDigis_vs_theta= new TH2F(N+"_hMuonTimeDigis_vs_theta","Muon digis time box vs theta",120,-60,60,960,0.,1500.);
    hMuonTimeDigis_vs_theta_RZ= new TH2F(N+"_hMuonTimeDigis_vs_theta_RZ","Muon digis time box vs theta only RZ SL",120,-60,60,960,0.,1500.); 
    hMuonTimeDigis_vs_theta_RPhi= new TH2F(N+"_hMuonTimeDigis_vs_theta_RPhi","Muon digis time box vs theta only RPhi SL",120,-60,60,960,0.,1500.);
  }

  virtual ~hDigis(){
//     delete hMuonDigis; 
//     delete hMuonTimeDigis; 
//     delete hMuonTimeDigis_vs_theta;
//     delete hMuonTimeDigis_vs_theta_RZ;
//     delete hMuonTimeDigis_vs_theta_RPhi;
  }
  	 
  hDigis(std::string name_,TFile *file){
    name=name_.c_str();
    //per lettura da file degli istogrammi
    //1D
    hMuonDigis = (TH1F *) file->Get(name+"_MuonDigis"); 
    hMuonTimeDigis  = (TH1F *) file->Get(name+"_MuonTimeDigis"); 
    //  control = (TH1F *) file->Get(name+"_control"); 
    //2D
    hMuonTimeDigis_vs_theta = (TH2F *) file->Get(name+"_MuonTimeDigis_vs_theta"); 
    hMuonTimeDigis_vs_theta_RZ= (TH2F *) file->Get(name+"_MuonTimeDigis_vs_theta_RZ"); 
    hMuonTimeDigis_vs_theta_RPhi= (TH2F *) file->Get(name+"_MuonTimeDigis_vs_theta_RPhi"); 
  }
  
  void Write(){
    // 1D
    hMuonDigis->Write(); 
    hMuonTimeDigis->Write(); 

    // 2D    
    hMuonTimeDigis_vs_theta->Write();
    hMuonTimeDigis_vs_theta_RZ->Write();
    hMuonTimeDigis_vs_theta_RPhi->Write();
  }
  
  //  void Fill(double ndigi, double time,double theta){
  void Fill(double time,double theta, int sltype){
    hMuonTimeDigis->Fill(time); 
    hMuonTimeDigis_vs_theta->Fill(theta,time);
    if (sltype==2) hMuonTimeDigis_vs_theta_RZ->Fill(theta,time);
    else hMuonTimeDigis_vs_theta_RPhi->Fill(theta,time);
  }

  void FillnDigis(int nDigis){
    hMuonDigis->Fill(nDigis);
  }

  void Asymmetry(){
    std::cout<<"["<<name<<"] theta asymmetry: "<<TH2asymmetry(hMuonTimeDigis_vs_theta)<<std::endl;
    std::cout<<"["<<name<<"] theta_RZ asymmetry: "<<TH2asymmetry(hMuonTimeDigis_vs_theta_RZ)<<std::endl;
    std::cout<<"["<<name<<"] theta_RPhi asymmetry: "<<TH2asymmetry(hMuonTimeDigis_vs_theta_RPhi)<<std::endl;
  }
  
 private:

  double TH2asymmetry(TH2F* H){
    double sx=H->Integral(0,60,0,960);
    double dx=H->Integral(60,120,0,960);
    double asym=(sx-dx)/(sx+dx);
    return asym;
  }
  


 public:
  //1D  
  TH1F* hMuonDigis; 
  TH1F* hMuonTimeDigis; 
  //2D
  TH2F *hMuonTimeDigis_vs_theta;
  TH2F *hMuonTimeDigis_vs_theta_RZ;
  TH2F *hMuonTimeDigis_vs_theta_RPhi;

 private:
  TString name;
   
};

class hHits{
 public:
  hHits(std::string name_){
    TString N = name_.c_str();
    name=N;
    //booking degli istogrammi unidimensionali
    hHitType= new TH1F(N+"_HitType","Hit Type distribution for "+N,20,-5,15);
    hDeltaZ = new TH1F(N+"_DeltaZ","z_{exit} - z_{entry} distribution for "+N,100,-1.2,1.2);
    hDeltaY = new TH1F(N+"_DeltaY","y_{exit} - y_{entry} distribution for "+N,100,-1.2,1.2);
    hDeltaX = new TH1F(N+"_DeltaX","x_{exit} - x_{entry} distribution for "+N,100,-1.2,1.2);
    hZentry = new TH1F(N+"_Zentry","z_{entry} distribution for "+N,500,-0.6,0.6);
    hZexit  = new TH1F(N+"_Zexit","z_{exit} distribution for "+N,500,-0.6,0.6);
    hXentry = new TH1F(N+"_Xentry","x_{entry} distribution for "+N,500,-0.6,0.6);
    hXexit  = new TH1F(N+"_Xexit","x_{exit} distribution for "+N,500,-0.6,0.6);
    hYentry = new TH1F(N+"_Yentry","y_{entry} distribution for "+N,500,-0.6,0.6);
    hYexit  = new TH1F(N+"_Yexit","y_{exit} distribution for "+N,500,-0.6,0.6);
    hHitMomentum = new TH1F(N+"_HitMomentum","Momentum distribution for "+N,100,0,100);
    hAbsZEntry   = new TH1F(N+"_AbsZEntry","|z| distribution for "+N+" entry points in the horizontal planes of the cell",100,0.57,0.58);
    hAbsZExit    = new TH1F(N+"_AbsZExit","|z| distribution for "+N+" exit points in the horizontal planes of the cell",100,0.57,0.58);
    hAbsXEntry   = new TH1F(N+"_AbsXEntry","|x| distribution for "+N+" entry points in the vertical planes of the cell",100,2.045,2.055);
    hAbsXExit    = new TH1F(N+"_AbsXExit","|x| distribution for "+N+" exit points in the vertical planes of the cell",100,2.04,2.06);
    hAbsYEntry   = new TH1F(N+"_AbsYEntry","|y| distribution for "+N+" entry points in the vertical planes of the cell",100,0,150);
    hAbsYExit    = new TH1F(N+"_AbsYExit","|y| distribution for "+N+" exit points in the vertical planes of the cell",100,0,150);
    hSagittaGeom = new TH1F(N+"_SagittaGeom","Geometric Sagitta distribution for "+N,100,0,.01);
    hSagittaMag= new TH1F(N+"_SagittaMag","Sagitta from magnetic bendig distribution, for "+N,100,0,.06);
    hSagittaPVSType= new TH2F(N+"_SagittaPVSType","Sagitta P VS hit type",14,0,14,100,0,.01);
    hSagittaBVSType= new TH2F(N+"_SagittaBVSType","Sagitta B VS hit type",14,0,14,100,0,.06);
    hPathVSType= new TH2F(N+"_PathVSType","Path VS hit type",14,0,14,840,0,4.2);
    hPathXVSType= new TH2F(N+"_PathXVSType","X Path VS hit type",14,0,14,840,0,4.2);
    hProcessType= new TH1F(N+"_ProcessType","Process Type",17,0,17);
    hProcessVsHitType = new TH2F(N+"_ProcessVsHitType","Process Type Vs Hit Type",14,0,14,17,0,17);
    hPathVsProcess = new TH2F(N+"_PathVsProcess","Path vs Process Type",14,0,14,840,0,4.2);
    hPathXVsProcess = new TH2F(N+"_PathXVsProcess","Path along X vs Process Type",14,0,14,840,0,4.2);

    h3DPathXVsProcessVsType = new TH3F(N+"_h3DPathXVsProcessVsType","Path along X vs Process Type and hit type",14,0,14,17,0,17,840,0,4.2);
    h3DPathVsProcessVsType  = new TH3F(N+"_h3DPathVsProcessVsType","Path vs Process Type and hit type",14,0,14,17,0,17,840,0,4.2);
    h3DXexitVsProcessVsType = new TH3F(N+"_h3DXexitVsProcessVsType","X exit vs Process Type and hit type",14,0,14,17,0,17,500,-0.6,0.6);

    hHitTOF= new TH1F(N+"_HitTOF","Hit TOF distribution for "+N,1000,1e4,1e8);

  }
  
  virtual ~hHits(){
//     delete hHitType;
//     delete hZentry;
//     delete hZexit;  
//     delete hXentry;
//     delete hXexit;
//     delete hYentry;
//     delete hYexit;
//     delete hDeltaZ;
//     delete hDeltaY;
//     delete hDeltaX;
//     delete hAbsZEntry;
//     delete hAbsZExit;
//     delete hAbsXEntry;
//     delete hAbsXExit;
//     delete hAbsYEntry;
//     delete hAbsYExit;
//     delete hHitMomentum;
//     delete hSagittaGeom;
//     delete hSagittaMag;
//     delete hSagittaPVSType;
//     delete hSagittaBVSType;
//     delete hPathVSType;
//     delete hPathXVSType;
//     delete hProcessType;
//     delete hProcessVsHitType;
//     delete hPathVsProcess;
//     delete hPathXVsProcess;

//     delete h3DPathXVsProcessVsType;
//     delete h3DPathVsProcessVsType;
//     delete h3DXexitVsProcessVsType;

 }
  	 
  hHits(std::string name_,TFile *file){
    name=name_.c_str();
    //per lettura da file degli istogrammi
    //1D
    hHitType =     (TH1F *) file->Get(name+"_HitType"); 
    hDeltaZ =      (TH1F *) file->Get(name+"_DeltaZ");
    hDeltaY =      (TH1F *) file->Get(name+"_DeltaY");
    hDeltaX =      (TH1F *) file->Get(name+"_DeltaX");
    hZentry =      (TH1F *) file->Get(name+"_Zentry");
    hZexit =       (TH1F *) file->Get(name+"_Zexit");
    hXentry =      (TH1F *) file->Get(name+"_Xentry");
    hXexit =       (TH1F *) file->Get(name+"_Xexit");
    hYentry =      (TH1F *) file->Get(name+"_Yentry");
    hYexit =       (TH1F *) file->Get(name+"_Yexit");
    hHitMomentum = (TH1F *) file->Get(name+"_HitMomentum");
    hAbsZEntry =   (TH1F *) file->Get(name+"_AbsZEntry");
    hAbsZExit =    (TH1F *) file->Get(name+"_AbsZExit");
    hAbsXEntry =   (TH1F *) file->Get(name+"_AbsXEntry");
    hAbsXExit =    (TH1F *) file->Get(name+"_AbsXExit");
    hAbsYEntry =   (TH1F *) file->Get(name+"_AbsYEntry");
    hAbsYExit =    (TH1F *) file->Get(name+"_AbsYExit");
    hSagittaGeom = (TH1F *) file->Get(name+"_SagittaGeom");
    hSagittaMag  = (TH1F *) file->Get(name+"_SagittaMag");
    hSagittaPVSType=(TH2F *) file->Get(name+"_SagittaPVSType");
    hSagittaBVSType=(TH2F *) file->Get(name+"_SagittaBVSType");
    hPathVSType    =(TH2F *) file->Get(name+"_PathVSType");
    hPathXVSType   =(TH2F *) file->Get(name+"_PathXVSType");
    hProcessType  = (TH1F *) file->Get(name+"_ProcessType");
    hProcessVsHitType = (TH2F *) file->Get(name+"_ProcessVsHitType");
    hPathVsProcess    = (TH2F *) file->Get(name+"_PathVsProcess");
    hPathXVsProcess   = (TH2F *) file->Get(name+"_PathXVsProcess");
    h3DPathXVsProcessVsType = (TH3F *) file->Get(name+"_h3DPathXVsProcessVsType");
    h3DPathVsProcessVsType  = (TH3F *) file->Get(name+"_h3DPathVsProcessVsType");
    h3DXexitVsProcessVsType = (TH3F *) file->Get(name+"_h3DXexitVsProcessVsType");

  }
  
  void Write(){
    hHitType->Write();
    hZentry->Write();
    hZexit->Write();
    hXentry->Write();
    hXexit->Write();
    hYentry->Write();
    hYexit->Write();
    hDeltaZ->Write();
    hDeltaY->Write();
    hDeltaX->Write();
    hAbsZEntry->Write();
    hAbsZExit->Write();
    hAbsXEntry->Write();
    hAbsXExit->Write();
    hAbsYEntry->Write();
    hAbsYExit->Write();
    hHitMomentum->Write();
    hSagittaGeom->Write();
    hSagittaMag->Write();
    hSagittaPVSType->Write();
    hSagittaBVSType->Write();
    hPathVSType->Write();
    hPathXVSType->Write();
    hProcessType->Write();
    hProcessVsHitType->Write();
    hPathVsProcess->Write();
    hPathXVsProcess->Write();
    h3DPathXVsProcessVsType->Write();
    h3DPathVsProcessVsType->Write();
    h3DXexitVsProcessVsType->Write();
    hHitTOF->Write();
  }
  
  void FillTOF(double tof){hHitTOF->Fill(tof);}

  void Fill(double xEntry,double xExit,
	    double entryPy, double exitPy,
	    double entryPz, double exitPz,
	    double path, double path_x,
	    int hitType,float processType,
	    double pAbs){//,double wire_length){

    hHitType->Fill(hitType);
    hZentry->Fill(entryPz);
    hZexit->Fill(exitPz);

    hXentry->Fill(xEntry);
    hXexit->Fill(xExit);
    hYentry->Fill(entryPy);
    hYexit->Fill(exitPy);

    hDeltaZ->Fill(exitPz - entryPz);
    hDeltaY->Fill(exitPy - entryPy);
    hDeltaX->Fill(xExit - xEntry);

    hHitMomentum->Fill(pAbs);

    hAbsZEntry->Fill(fabs(entryPz));
    hAbsZExit->Fill(fabs(exitPz));

    if(fabs(entryPz) < 0.573) {
      hAbsXEntry->Fill(fabs(xEntry));  
    }

    if(fabs(exitPz) < 0.573) {
      hAbsXExit->Fill(fabs(xExit));
    }

    //  hAbsYEntry->Fill(wire_length/2.-fabs(entryPy)); 
    // hAbsYExit->Fill(wire_length/2.-fabs(exitPy)); 

    hPathVSType->Fill(hitType,path);
    hPathXVSType->Fill(hitType,path_x);
    hProcessType->Fill(processType);
    hProcessVsHitType->Fill(hitType,processType);
    hPathVsProcess->Fill(processType,path);
    hPathXVsProcess->Fill(processType,path_x);

    h3DPathXVsProcessVsType->Fill(processType,hitType,path_x);
    h3DPathVsProcessVsType->Fill(processType,hitType,path);
    h3DXexitVsProcessVsType->Fill(processType,hitType,xExit);
}

  void FillSagittas(double SG,double SM,int hitType){
    hSagittaGeom->Fill(SG);
    hSagittaMag->Fill(SM);
    hSagittaPVSType->Fill(hitType,SG);
    hSagittaBVSType->Fill(hitType,SM);
  }

 public:
  TH1F *hHitType;
  TH1F *hZentry;
  TH1F *hZexit;
  TH1F *hXentry;
  TH1F *hXexit;
  TH1F *hYentry;
  TH1F *hYexit;
  TH1F *hDeltaZ;
  TH1F *hDeltaY;
  TH1F *hDeltaX;
  TH1F *hAbsZEntry;
  TH1F *hAbsZExit;
  TH1F *hAbsXEntry;
  TH1F *hAbsXExit;
  TH1F *hAbsYEntry;
  TH1F *hAbsYExit;
  TH1F *hHitMomentum;
  TH1F *hSagittaGeom; 
  TH1F *hSagittaMag;
  TH2F *hSagittaPVSType;
  TH2F *hSagittaBVSType;
  TH2F *hPathVSType;
  TH2F *hPathXVSType;
  TH1F *hProcessType;
  TH2F *hProcessVsHitType;
  TH2F *hPathVsProcess;
  TH2F *hPathXVsProcess;
  TH3F *h3DPathXVsProcessVsType;
  TH3F *h3DPathVsProcessVsType;
  TH3F *h3DXexitVsProcessVsType;
  TH1F *hHitTOF;

 private:
  TString name;
};

class hDeltaR{
 public:
  hDeltaR(std::string name_){
    TString N = name_.c_str();
    name=N;
    //booking degli istogrammi unidimensionali
    hZentry = new TH1F(N+"_Zentry","z_{entry} distribution",120,-0.6,0.6);
    hXentry = new TH1F(N+"_Xentry","x_{entry} distribution",120,-4.2,4.2);
    hYentry = new TH1F(N+"_Yentry","y_{entry} distribution",120,-400,400);
    hHitMomentum = new TH1F(N+"_HitMomentum","Momentum distribution",100,0,.2);
    hHitEnergyLoss = new TH1F(N+"_HitEnergyLoss","Energy Loss distribution",75,0,100); //in keV--> x10^6
    hSagittaMag= new TH1F(N+"_SagittaMag","Sagitta from magnetic bendig",120,0,.04);
    hPath= new TH1F(N+"_Path","Path",200,0,4.2);
    hPathX= new TH1F(N+"_PathX","X Path",200,0,4.2);
    hZoomPath= new TH1F(N+"_ZoomPath","Path",200,0,.1);
    hZoomPathX= new TH1F(N+"_ZoomPathX","X Path",200,0,.1);
    hType = new TH1F(N+"_Type","Delta type 3-electron 2-positron",4,0,4);
  }
   
  virtual ~hDeltaR(){
//     delete hType;
//     delete hZentry;
//     delete hXentry;
//     delete hYentry;
//     delete hHitMomentum;
//     delete hHitEnergyLoss;
//     delete hSagittaMag;
//     delete hPath;
//     delete hPathX;
//     delete hZoomPath;
//     delete hZoomPathX;
  }
  
  hDeltaR(std::string name_,TFile *file){
    name=name_.c_str();
    //per lettura da file degli istogrammi
    //1D
    hType =      (TH1F *) file->Get(name+"_Type");
    hZentry =      (TH1F *) file->Get(name+"_Zentry");
    hXentry =      (TH1F *) file->Get(name+"_Xentry");
    hYentry =      (TH1F *) file->Get(name+"_Yentry");
    hHitMomentum = (TH1F *) file->Get(name+"_HitMomentum");
    hHitMomentum = (TH1F *) file->Get(name+"_HitEnergyLoss");
    hSagittaMag  = (TH1F *) file->Get(name+"_SagittaMag");
    hPath    =(TH1F *) file->Get(name+"_Path");
    hPathX   =(TH1F *) file->Get(name+"_PathX");
    hZoomPath    =(TH1F *) file->Get(name+"_ZoomPath");
    hZoomPathX   =(TH1F *) file->Get(name+"_ZoomPathX");
  }
  
  void Write(){
    hType->Write();
    hZentry->Write();
    hXentry->Write();
    hYentry->Write();
    hHitMomentum->Write();
    hSagittaMag->Write();
    hPath->Write();
    hPathX->Write();
    hZoomPath->Write();
    hZoomPathX->Write();

  }
  
  void Fill(double xEntry, double yEntry, double zEntry,
	    double path, double path_x, double SM,
	    double pAbs, double En, int type){
    
    hZentry->Fill(zEntry);
    hXentry->Fill(xEntry);
    hYentry->Fill(yEntry);

    hHitMomentum->Fill(pAbs);
    hHitEnergyLoss->Fill(En*1e6);
    
    hPath->Fill(path);
    hPathX->Fill(path_x);
    if(path<=.1) hZoomPath->Fill(path);
    if(path_x<=.1) hZoomPathX->Fill(path_x);
    
    hSagittaMag->Fill(SM);
    hType->Fill(type);
  }
  
 public:
  TH1F *hType;
  TH1F *hZentry;
  TH1F *hXentry;
  TH1F *hYentry;
  TH1F *hHitMomentum;		
  TH1F *hHitEnergyLoss;
  TH1F *hSagittaMag;
  TH1F *hPath;
  TH1F *hPathX;
  TH1F *hZoomPath;
  TH1F *hZoomPathX;

  
 private:
  TString name;
};


class hParam{
 public:
  hParam(std::string name_){
    TString N = name_.c_str();
    name=N;
    //booking degli istogrammi unidimensionali
    HitParam_X = new TH1F(N+"_HitParam_X","Distribution of theta for parameterization cases in Rphi layers",100,-2.1,2.1);  
    HitParam_Theta = new TH1F(N+"_HitParam_Theta","Distribution of theta for parameterization cases in Rphi layers",100,-180.,180.);  
    HitParam_Bwire = new TH1F(N+"_HitParam_Bwire","Distribution of bwire for parameterization cases in Rz layers",100,-0.5,0.5);
    HitParam_Bnorm = new TH1F(N+"_HitParam_Bnorm","Distribution of bnorm for parameterization cases in Rphi layers",100,-1,1);  
  }
  
  virtual ~hParam(){
//   delete HitParam_X;
//   delete HitParam_Theta;
//   delete HitParam_Bwire;
//   delete HitParam_Bnorm;
  }
  	 
  hParam(std::string name_,TFile *file){
    name=name_.c_str();
    //per lettura da file degli istogrammi
    //1D
    HitParam_X       = (TH1F *) file->Get(name+"_HitParam_X");
    HitParam_Theta   = (TH1F *) file->Get(name+"_HitParam_Theta");
    HitParam_Bwire   = (TH1F *) file->Get(name+"_HitParam_Bwire");
    HitParam_Bnorm   = (TH1F *) file->Get(name+"_HitParam_Bnorm");
  }

  void Fill(double x,double theta,double Bwire,double Bnorm){
    HitParam_X->Fill(x);
    HitParam_Theta->Fill(theta);
    HitParam_Bwire->Fill(Bwire);
    HitParam_Bnorm->Fill(Bnorm);
  }

  void Write(){
    HitParam_X->Write();
    HitParam_Theta->Write();
    HitParam_Bwire->Write();
    HitParam_Bnorm->Write();
  }
  
 public:
  TH1F *HitParam_X;
  TH1F *HitParam_Theta;
  TH1F *HitParam_Bwire;
  TH1F *HitParam_Bnorm;

 private:
  TString name;

};

class hMuonStat{
 public:
    hMuonStat(std::string name_){
      TString N = name_.c_str();
      name=N;
      //booking degli istogrammi unidimensionali
      hMuonNumber  = new TH1F ("hMuon"+N, "Muon hits ", 200, 0., 200.);
      hMuonVsEta  = new TH1F ("hMuon"+N+"VsEta", "Muon "+N+" vs eta",100, -1.2, 1.2);
      hMuonVsPhi  = new TH1F ("hMuon"+N+"VsPhi", "Muon "+N+" vs phi",100, -Geom::pi(), +Geom::pi());
    }
        
    hMuonStat(std::string name_,TFile *file){
      name=name_.c_str();
      //per lettura da file degli istogrammi
      //1D
      hMuonNumber  =  (TH1F *) file->Get("hMuon"+name);
      hMuonVsEta =  (TH1F *) file->Get("hMuon"+name+"VsEta");
      hMuonVsPhi  = (TH1F *) file->Get("hMuon"+name+"VsPhi");
    }
    
    ~hMuonStat(){
//       delete hMuonNumber;
//       delete hMuonVsEta;
//       delete hMuonVsPhi;
    }
    void Write(){
      hMuonNumber->Write();
      hMuonVsEta->Write();
      hMuonVsPhi->Write();
    }

    void Fill(int hits,double eta, double phi){
      hMuonNumber->Fill(hits);
      hMuonVsEta->Fill(eta,hits);
      hMuonVsPhi->Fill(phi,hits);
    }

 public:
    TH1F* hMuonNumber;
    TH1F* hMuonVsEta;
    TH1F* hMuonVsPhi;
    
 private:
  TString   name;

};


class hTOF{
public:
  hTOF(std::string name){
    TString N = name.c_str();

    hTOF_true =   new TH1F(N+"_TOF_true","TOF true",200,10.,35.);
    hTOF_hitPos = new TH1F(N+"_TOF_hitPos","TOF assumed, hit pos",200,10.,35.);
    hTOF_WC     = new TH1F(N+"_TOF_WC","TOF assumed, wire center",200,10.,35.);

    hDeltaTOF_hitPos = new TH1F(N+"_DeltaTOF_hitPos","TOF assumed, hit pos",200,-5.,5.);
    hDeltaTOF_WC     = new TH1F(N+"_DelataTOF_WC","TOF assumed, wire center",200,-5.,5.);

  }
  
  hTOF(std::string name_,TFile *file){
    name=name_.c_str();
    hTOF_true =  (TH1F *) file->Get(name+"_TOF_true");
    hTOF_hitPos=  (TH1F *) file->Get(name+"_TOF_hitPos");
    hTOF_WC    =  (TH1F *) file->Get(name+"_TOF_WC");
    hDeltaTOF_hitPos= (TH1F *) file->Get(name+"_DeltaTOF_hitPos");
    hDeltaTOF_WC=     (TH1F *) file->Get(name+"_DeltaTOF_WC");
  }


  ~hTOF(){
//     delete hTOF_true;
//     delete hTOF_hitPos;
//     delete hTOF_WC;
//     delete hDeltaTOF_hitPos;
//     delete hDeltaTOF_WC;
  }
  
  void Fill(float tof, float TOF_WC, float TOF_hitPos){
    hTOF_hitPos->Fill(TOF_hitPos);
    hTOF_WC->Fill(TOF_WC);
    hTOF_true->Fill(tof);
    hDeltaTOF_hitPos->Fill(tof-TOF_hitPos);
    hDeltaTOF_WC->Fill(tof-TOF_WC);
  }


  void Write(){
    hTOF_true->Write();
    hTOF_hitPos->Write();
    hTOF_WC->Write();
    hDeltaTOF_hitPos->Write();
    hDeltaTOF_WC->Write();
  }


public:
  TH1F * hTOF_true;
  TH1F * hTOF_hitPos;
  TH1F * hTOF_WC;
  TH1F * hDeltaTOF_hitPos;
  TH1F * hDeltaTOF_WC;

 private:
  TString name;
  
};

class hTDelay{
public:
  hTDelay(std::string name){
    TString N = name.c_str();
    
    hTDelay_true     = new TH1F(N+"_TDelay_true", "Delay (true)",
			   100, 0., 15.);
    hTDelay_WC  = new TH1F(N+"_TDelay_WC", "Delay (assumed, wire center)",
			   100, 0., 15.);

    hTDelay_hitpos  = new TH1F(N+"_TDelay_hitpos", "Delay (assumed, hit pos)",
			       100, 0., 15.);
    
    hDeltaTDelay_WC = new TH1F(N+"_dTDelay_WC", "Delay true - WC",
			   150, -15, 15.);
    hDeltaTDelay_hitpos = new TH1F(N+"_dTDelay_hitpos", "Delay true - hitpos",
			       150, -15, 15.);

  }
  
  hTDelay(std::string name_,TFile *file){
    name=name_.c_str();
    hTDelay_true = (TH1F *) file->Get(name+"_TDelay_true");
    hTDelay_WC  = (TH1F *) file->Get(name+"_TDelay_WC");
    hTDelay_hitpos  = (TH1F *) file->Get(name+"_TDelay_hitpos");
    hDeltaTDelay_WC = (TH1F *) file->Get(name+"_DeltaTDelay_WC");
    hDeltaTDelay_hitpos = (TH1F *) file->Get(name+"_DeltaTDelay_hitpos");
  }
  ~hTDelay() {
//     delete hTDelay_true;	
//     delete hTDelay_WC;	
//     delete hTDelay_hitpos;
//     delete hDeltaTDelay_WC;
//     delete hDeltaTDelay_hitpos;
  }  

  void Fill(float t_True, float t_WC, float t_hitpos){
   hTDelay_true->Fill(t_True);	  
   hTDelay_WC->Fill(t_WC);  
   hTDelay_hitpos->Fill(t_hitpos);
   hDeltaTDelay_WC->Fill(t_WC-t_True);
   hDeltaTDelay_hitpos->Fill(t_hitpos-t_True);
  }

  void Write(){
    hTDelay_true->Write();	
    hTDelay_WC->Write();
    hTDelay_hitpos->Write();
    hDeltaTDelay_WC->Write();
    hDeltaTDelay_hitpos->Write();
  }  

public:
  TH1F * hTDelay_true;
  TH1F * hTDelay_WC;
  TH1F * hTDelay_hitpos;
  TH1F * hDeltaTDelay_WC;
  TH1F * hDeltaTDelay_hitpos;
 private:
  TString name;
};

template<class hTime>
class hTimes{
 public:
  hTimes(std::string name){
    RZ = new hTime(name+"_RZ");
    RPhi = new hTime(name+"_RPhi");
    W0 = new hTime(name+"_Wheel0");
    W1 = new hTime(name+"_Wheel1");
    W2 = new hTime(name+"_Wheel2");  
  }
  
  hTimes(std::string name_,TFile *file){
    name=name_.c_str();

    RZ =   (hTime *) file->Get(name+"_RZ");
    RPhi = (hTime *) file->Get(name+"_RPhi");
    W0 =   (hTime *) file->Get(name+"_Wheel0");
    W1 =   (hTime *) file->Get(name+"_Wheel1");
    W2 =   (hTime *) file->Get(name+"_Wheel2");
  }

 
  ~hTimes(){
//     delete RZ;
//     delete RPhi;
//     delete W0;
//     delete W1;
//     delete W2;
  }

  void Fill(float t_True, float t_WC, float t_hitpos, int wheel_type, int sltype){
    if (sltype==2) RZ->Fill(t_True,t_WC,t_hitpos);
    else  RPhi->Fill(t_True,t_WC,t_hitpos);
    WheelHistos(wheel_type)->Fill(t_True,t_WC,t_hitpos);
  }
  
  void Write(){
    RZ->Write();
    RPhi->Write();
    W0->Write();
    W1->Write();
    W2->Write();
  }  


  hTime* WheelHistos(int wheel){
  switch(abs(wheel)){

  case 0: return  W0;
  
  case 1: return  W1;
    
  case 2: return  W2;
     
  default: return NULL;
  }
}
 private:
  hTime *RZ;
  hTime *RPhi;
  hTime *W0;
  hTime *W1;
  hTime *W2;
 private:
  TString name;
};

#endif

