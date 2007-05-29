#ifndef Validation_RecoMuon_Histograms_H
#define Validation_RecoMuon_Histograms_H

/** \class Histograms
 *  No description available.
 *
 *  $Date: 2007/01/24 10:30:21 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "TH1F.h"
#include "TH2F.h"
#include "TString.h"
#include "TFile.h"

#include "DataFormats/GeometryVector/interface/Pi.h"
#include <iostream>

class HTrackVariables{
public:
  
  HTrackVariables(std::string name,std::string whereIs =""):theName(name.c_str()),where(whereIs.c_str()){

    hEta = new TH1F(theName+"_Eta_"+where,"#eta at "+where,120,-3.,3.);
    hPhi = new TH1F(theName+"_Phi_"+where,"#phi at "+where,100,-Geom::pi(),Geom::pi());
    hP   = new TH1F(theName+"_P_"+where,"p_{t} at "+where,1000,0,2000);
    hPt  = new TH1F(theName+"_Pt_"+where,"p_{t} at "+where,1000,0,2000);
    hCharge = new TH1F(theName+"_charge_"+where,"Charge at "+where,4,-2.,2.);

    hEtaVsGen = new TH1F(theName+"_EtaVsGen_"+where,"#eta at "+where,120,-3.,3.);
    hPhiVsGen = new TH1F(theName+"_PhiVsGen_"+where,"#phi at "+where,100,-Geom::pi(),Geom::pi());
    hPtVsGen  = new TH1F(theName+"_PtVsGen_"+where,"p_{t} at "+where,1000,0,2000);

    hDeltaR = new TH1F(theName+"_DeltaR_"+where,"Delta R w.r.t. sim track for "+where,1000,0,20);

    theEntries = 0;
  }
  
  HTrackVariables(std::string name, TFile* file):theName(name.c_str()){ 
    
    hEta = dynamic_cast<TH1F*>( file->Get(theName+"_Eta_"+where));
    hPhi = dynamic_cast<TH1F*>( file->Get(theName+"_Phi_"+where));
    hP   = dynamic_cast<TH1F*>( file->Get(theName+"_P_"+where));
    hPt  = dynamic_cast<TH1F*>( file->Get(theName+"_Pt_"+where));
    hCharge = dynamic_cast<TH1F*>( file->Get(theName+"_charge_"+where)); 
  }
  
  ~HTrackVariables(){}

  TH1F *eta() {return hEta;}
  TH1F *phi() {return hPhi;}
  TH1F *p() {return hP;}
  TH1F *pt() {return hPt;}
  TH1F *charge() {return hCharge;}
  int entries() {return theEntries;}

  void Fill(double p, double pt, double eta, double phi, double charge){
    hEta->Fill(eta);
    hPhi->Fill(phi);
    hP->Fill(p);
    hPt->Fill(pt);
    hCharge->Fill(charge);
    ++theEntries;
  }

  void Fill(double pt, double eta, double phi){
    hEtaVsGen->Fill(eta);
    hPhiVsGen->Fill(phi);
    hPtVsGen->Fill(pt);
  }

  void FillDeltaR(double deltaR){
    hDeltaR->Fill(deltaR);
  }



  void Write(){
    hEta->Write();
    hPhi->Write();
    hP->Write();
    hPt->Write();
    hCharge->Write();

    hDeltaR->Write();
    // hEtaVsGen->Write();
    // hPhiVsGen->Write();
    // hPtVsGen->Write();

    for(std::vector<TH1F*>::iterator histo = efficiencyHistos.begin(); 
	histo != efficiencyHistos.end(); ++histo)
      (*histo)->Write();
  }

  double computeEfficiencyAndWrite(HTrackVariables *sim){
    
    computeEfficiency(sim);
    Write();
    double efficiency = 100*entries()/sim->entries();
    return efficiency;
  }
  
  double computeEfficiency(HTrackVariables *sim){
    
    efficiencyHistos.push_back(computeEfficiency(hEtaVsGen,sim->eta()));
    efficiencyHistos.push_back(computeEfficiency(hPhiVsGen,sim->phi()));
    //    efficiencyHistos.push_back(computeEfficiency(p(),sim->p()));
    efficiencyHistos.push_back(computeEfficiency(hPtVsGen,sim->pt()));
    //    efficiencyHistos.push_back(computeEfficiency(charge(),sim->charge()));

    double efficiency = 100*entries()/sim->entries();
    return efficiency;
  }

  TH1F* computeEfficiency(TH1F *reco, TH1F *sim){
    
    TH1F *hEff = (TH1F*) reco->Clone();
    
    TString name = hEff->GetName();
    TString title = hEff->GetTitle();
    
    hEff->SetName("Eff_"+name);
    hEff->SetTitle("Efficiency as a function of "+title);
    
    hEff->Divide(sim);
    
    // Set the error accordingly to binomial statistics
    int nBinsEta = hEff->GetNbinsX();
    for(int bin = 1; bin <=  nBinsEta; bin++) {
      float nSimHit = sim->GetBinContent(bin);
      float eff = hEff->GetBinContent(bin);
      float error = 0;
      if(nSimHit != 0 && eff <= 1) {
        error = sqrt(eff*(1-eff)/nSimHit);
      }
      hEff->SetBinError(bin, error);
    }
    
    return hEff;
  }


private:
  TString theName;
  TString where;
  
  int theEntries;

  TH1F *hEta;
  TH1F *hPhi;
  TH1F *hP;
  TH1F *hPt;
  TH1F *hCharge;

  TH1F *hEtaVsGen;
  TH1F *hPhiVsGen;
  TH1F *hPtVsGen;
  
  TH1F* hDeltaR;
  
  std::vector<TH1F*> efficiencyHistos;

  
};


class HResolution {
public:
  
  HResolution(std::string name,std::string whereIs):theName(name.c_str()),where(whereIs.c_str()){
 
    double eta = 15.; int neta = 800;
    double phi = 12.; int nphi = 400;
    double pt = 60.; int npt = 2000;

    hEta = new TH1F(theName+"_Eta_"+where,"#eta "+theName,neta,-eta,eta); // 400
    hPhi = new TH1F(theName+"_Phi_"+where,"#phi "+theName,nphi,-phi,phi); // 100

    hP = new TH1F(theName+"_P_"+where,"P "+theName,400,-4,4);  // 200
    hPt = new TH1F(theName+"_Pt_"+where,"P_{t} "+theName,npt,-pt,pt); // 200

    hCharge = new TH1F(theName+"_charge_"+where,"Charge "+theName,4,-2.,2.);


    h2Eta = new TH2F(theName+"_Eta_vs_Eta"+where,"#eta "+theName+" as a function of #eta",200,-2.5,2.5,neta,-eta,eta);
    h2Phi = new TH2F(theName+"_Phi_vs_Phi"+where,"#phi "+theName+" as a function of #phi",100,-Geom::pi(),Geom::pi(),nphi,-phi,phi);
    
    h2P = new TH2F(theName+"_P_vs_P"+where,"P "+theName+" as a function of P",1000,0,2000,400,-4,4);
    h2Pt = new TH2F(theName+"_Pt_vs_Pt"+where,"P_{t} "+theName+" as a function of P_{t}",1000,0,2000,npt,-pt,pt);
    
    h2PtVsEta = new TH2F(theName+"_Pt_vs_Eta"+where,"P_{t} "+theName+" as a function of #eta",200,-2.5,2.5,npt,-pt,pt);
    h2PtVsPhi = new TH2F(theName+"_Pt_vs_Phi"+where,"P_{t} "+theName+" as a function of #phi",100,-Geom::pi(),Geom::pi(),npt,-pt,pt);

    h2EtaVsPt = new TH2F(theName+"_Eta_vs_Pt"+where,"#eta "+theName+" as a function of P_{t}",1000,0,2000,neta,-eta,eta);
    h2EtaVsPhi = new TH2F(theName+"_Eta_vs_Phi"+where,"#eta "+theName+" as a function of #phi",100,-Geom::pi(),Geom::pi(),neta,-eta,eta);
    
    h2PhiVsPt = new TH2F(theName+"_Phi_vs_Pt"+where,"#phi "+theName+" as a function of P_{t}",1000,0,2000,nphi,-phi,phi);
    h2PhiVsEta = new TH2F(theName+"_Phi_vs_Eta"+where,"#phi "+theName+" as a function of #eta",200,-2.5,2.5,nphi,-phi,phi);
  }
  
  HResolution(std::string name, TFile* file):theName(name.c_str()){ 
    //    dynamic_cast<TH1F*>( file->Get(theName+"") );
  }
  
  ~HResolution(){}


  void Fill(double p, double pt, double eta, double phi,
	    double rp, double rpt,double reta, double rphi, double rcharge){
   
    Fill(rp, rpt, reta, rphi, rcharge);

    
    h2Eta->Fill(eta,reta);
    h2Phi->Fill(phi,rphi);
    
    h2P->Fill(p,rp); 
    h2Pt->Fill(pt,rpt);

    h2PtVsEta->Fill(eta,rpt);
    h2PtVsPhi->Fill(phi,rpt);
    
    h2EtaVsPt ->Fill(pt,reta);
    h2EtaVsPhi->Fill(phi,reta);
    
    h2PhiVsPt ->Fill(pt,rphi);
    h2PhiVsEta->Fill(eta,rphi);
  }


  void Fill(double p, double pt, double eta, double phi,
	    double rp, double rpt){
   
    hP->Fill(rp); 
    hPt->Fill(rpt);
    
    h2P->Fill(p,rp); 
    // h2PVsEta->Fill(eta,rp);
    // h2PVsPhi->Fill(phi,rp);

    h2Pt->Fill(pt,rpt);
    h2PtVsEta->Fill(eta,rpt);
    h2PtVsPhi->Fill(phi,rpt);
  }
    
  void Fill(double rp, double rpt, 
	    double reta, double rphi, double rcharge){
    
    hEta->Fill(reta);
    hPhi->Fill(rphi);
    
    hP->Fill(rp); 
    hPt->Fill(rpt);
    
    hCharge->Fill(rcharge);
  }

  void Write(){
    hEta->Write();
    hPhi->Write();
    
    hP->Write(); 
    hPt->Write(); 
    
    hCharge->Write();

    h2Eta->Write();
    h2Phi->Write();
    
    h2P->Write(); 
    h2Pt->Write(); 

    h2PtVsEta->Write();
    h2PtVsPhi->Write();

    h2EtaVsPt->Write();
    h2EtaVsPhi->Write();
    
    h2PhiVsPt ->Write();
    h2PhiVsEta->Write();
  }

private:
  TString theName;
  TString where;

  TH1F *hEta;
  TH1F *hPhi;

  TH1F *hP; 
  TH1F *hPt; 

  TH1F *hCharge;

  TH2F *h2Eta;
  TH2F *h2Phi;

  TH2F *h2P; 
  TH2F *h2Pt; 

  TH2F *h2PtVsEta;
  TH2F *h2PtVsPhi;

  TH2F *h2EtaVsPt;
  TH2F *h2EtaVsPhi;
  
  TH2F *h2PhiVsPt; 
  TH2F *h2PhiVsEta;

};


class HResolution1DRecHit{
 public:
  HResolution1DRecHit(std::string name):theName(name.c_str()){

    // Position, sigma, residual, pull
    hResX        = new TH1F (theName+"_X_Res", "X residual", 5000, -0.5,0.5);
    hResY        = new TH1F (theName+"_Y_Res", "Y residual", 5000, -1.,1.);
    hResZ        = new TH1F (theName+"_Z_Res", "Z residual", 5000, -1.5,1.5);

    hResXVsEta   = new TH2F(theName+"_XResVsEta", "X residual vs eta",
			    200, -2.5,2.5, 5000, -1.5,1.5);
    hResYVsEta   = new TH2F(theName+"_YResVsEta", "Y residual vs eta",
			    200, -2.5,2.5, 5000, -1.,1.);
    hResZVsEta   = new TH2F(theName+"_ZResVsEta", "Z residual vs eta",
			    200, -2.5,2.5, 5000, -1.5,1.5);
    
    hXResVsPhi   = new TH2F(theName+"_XResVsPhi", "X residual vs phi",
			    100,-Geom::pi(),Geom::pi(), 5000, -0.5,0.5);
    hYResVsPhi   = new TH2F(theName+"_YResVsPhi", "Y residual vs phi",
			    100,-Geom::pi(),Geom::pi(), 5000, -1.,1.);
    hZResVsPhi   = new TH2F(theName+"_ZResVsPhi", "Z residual vs phi",
			    100,-Geom::pi(),Geom::pi(), 5000, -1.5,1.5);
    
    hXResVsPos   = new TH2F(theName+"_XResVsPos", "X residual vs position",
			    10000, -750,750, 5000, -0.5,0.5);    
    hYResVsPos   = new TH2F(theName+"_YResVsPos", "Y residual vs position",
			    10000, -740,740, 5000, -1.,1.);    
    hZResVsPos   = new TH2F(theName+"_ZResVsPos", "Z residual vs position",
			    10000, -1100,1100, 5000, -1.5,1.5);   
    
    hXPull       = new TH1F (theName+"_XPull", "X pull", 600, -2,2);
    hYPull       = new TH1F (theName+"_YPull", "Y pull", 600, -2,2);
    hZPull       = new TH1F (theName+"_ZPull", "Z pull", 600, -2,2);

    hXPullVsPos  = new TH2F (theName+"_XPullVsPos", "X pull vs position",10000, -750,750, 600, -2,2);
    hYPullVsPos  = new TH2F (theName+"_YPullVsPos", "Y pull vs position",10000, -740,740, 600, -2,2);
    hZPullVsPos  = new TH2F (theName+"_ZPullVsPos", "Z pull vs position",10000, -1100,1100, 600, -2,2);
  }
  
  HResolution1DRecHit(TString name_, TFile* file){}

  ~HResolution1DRecHit(){}

  void Fill(double x, double y, double z,
	    double dx, double dy, double dz,
	    double errx, double erry, double errz,
	    double eta, double phi) {
    
    double rx = dx/x, ry = dy/y, rz = dz/z;

    hResX->Fill(rx);
    hResY->Fill(ry);
    hResZ->Fill(rz);
    
    hResXVsEta->Fill(eta,rx);
    hResYVsEta->Fill(eta,ry);
    hResZVsEta->Fill(eta,rz);
    
    hXResVsPhi->Fill(phi,rx);
    hYResVsPhi->Fill(phi,ry);
    hZResVsPhi->Fill(phi,rz);
    
    hXResVsPos->Fill(x,rx);
    hYResVsPos->Fill(y,ry);
    hZResVsPos->Fill(z,rz);
    
    if(errx < 1e-6)
      std::cout << "NO proper error set for X: "<<errx<<std::endl;
    else{
      hXPull->Fill(dx/errx);
      hXPullVsPos->Fill(x,dx/errx);
    }
    
    if(erry < 1e-6) 
      std::cout << "NO proper error set for Y: "<<erry<<std::endl;
    else{
      hYPull->Fill(dy/erry);
      hYPullVsPos->Fill(y,dy/erry);
    }
    if(errz < 1e-6)
      std::cout << "NO proper error set for Z: "<<errz<<std::endl; 
    else{
      hZPull->Fill(dz/errz);
      hZPullVsPos->Fill(z,dz/errz);
    }
  }
  
  void Write() {
 
    hResX->Write();
    hResY->Write();
    hResZ->Write();
    
    hResXVsEta->Write();
    hResYVsEta->Write();
    hResZVsEta->Write();
  
    hXResVsPhi->Write();
    hYResVsPhi->Write();
    hZResVsPhi->Write();
    
    hXResVsPos->Write();
    hYResVsPos->Write();
    hZResVsPos->Write();
    
    hXPull->Write();
    hYPull->Write();
    hZPull->Write();
    
    hXPullVsPos->Write();
    hYPullVsPos->Write();
    hZPullVsPos->Write();
  }

  
 public:
  TString theName;
  
  TH1F *hResX;
  TH1F *hResY;
  TH1F *hResZ;
  
  TH2F *hResXVsEta;
  TH2F *hResYVsEta;
  TH2F *hResZVsEta;
  
  TH2F *hXResVsPhi;
  TH2F *hYResVsPhi;
  TH2F *hZResVsPhi;
  
  TH2F *hXResVsPos;
  TH2F *hYResVsPos;
  TH2F *hZResVsPos;
  
  TH1F *hXPull;
  TH1F *hYPull;
  TH1F *hZPull;
  
  TH2F *hXPullVsPos;
  TH2F *hYPullVsPos;
  TH2F *hZPullVsPos;
};
#endif

