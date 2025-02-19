#include "TFile.h"
#include "TTree.h"
#include "TText.h"

void ECALMaterialBudgetCompare()
{

 gROOT ->Reset();
 char*  rfilename = "old_matbdg_ECAL.root";
 char*  sfilename = "new_matbdg_ECAL.root";

 int rcolor = 2;
 int scolor = 4;

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename);

 TText* te = new TText();
 te->SetTextSize(0.1);
 TFile * rfile = new TFile(rfilename);
 TFile * sfile = new TFile(sfilename);

 Char_t histo[200];

 gStyle->SetOptStat("n");


// Global Eta and Phi plots
 
 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);
   
   TH1* eta_;
   rfile->GetObject("10;1",eta_);
   eta_;
   eta_->SetLineColor(rcolor);
   
   TH1* neweta_;
   sfile->GetObject("10;1",neweta_);
   neweta_;
   neweta_->SetLineColor(scolor);
   
   TH1* phi_;
   rfile->GetObject("20;1",phi_);
   phi_;
   phi_->SetLineColor(rcolor);
   
   TH1* newphi_;
   sfile->GetObject("20;1",newphi_);
   newphi_;
   newphi_->SetLineColor(scolor);
   
   Ecal->cd(1); 
   eta_->Draw(); 
   neweta_->Draw("same");
   Ecal->cd(2); 
   phi_->Draw(); 
   newphi_->Draw("same");
   Ecal->Print("Global_MB_compare.eps"); 
 }

 // ECAL barrel plots

 
  if (1) { 
    TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000); 
    Ecal->Divide(1,3); 
 
 
    TH1* eta_; 
    rfile->GetObject("1001;1",eta_); 
    eta_; 
    eta_->SetLineColor(rcolor); 
 
    TH1* neweta_; 
    sfile->GetObject("1001;1",neweta_); 
    neweta_; 
    neweta_->SetLineColor(scolor); 
 
    TH1* phi_; 
    rfile->GetObject("1002;1",phi_); 
    phi_; 
    phi_->SetLineColor(rcolor); 
 
    TH1* newphi_; 
    sfile->GetObject("1002;1",newphi_); 
    newphi_; 
    newphi_->SetLineColor(scolor); 
 
    TH1* smphi_; 
    rfile->GetObject("1003;1",smphi_); 
    smphi_; 
    smphi_->SetLineColor(rcolor); 
 
    TH1* newsmphi_; 
    sfile->GetObject("1003;1",newsmphi_); 
    newsmphi_; 
    newsmphi_->SetLineColor(scolor); 
 
    Ecal->cd(1);  
    eta_->Draw();  
    neweta_->Draw("same"); 
    Ecal->cd(2);  
    phi_->Draw();  
    newphi_->Draw("same"); 
    Ecal->cd(3);  
    smphi_->Draw();  
    newsmphi_->Draw("same"); 
    Ecal->Print("ECAL_Barrel_MB_compare.eps");  

  }  
 
  if (1) { 
    TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000); 
    Ecal->Divide(2,2); 
   
   
    TH1* phimod1_; 
    rfile->GetObject("1004;1",phimod1_); 
    phimod1_; 
    phimod1_->SetLineColor(rcolor); 
   
    TH1* newphimod1_; 
    sfile->GetObject("1004;1",newphimod1_); 
    newphimod1_; 
    newphimod1_->SetLineColor(scolor); 
   
    TH1* phimod2_; 
    rfile->GetObject("1005;1",phimod2_); 
    phimod2_; 
    phimod2_->SetLineColor(rcolor); 
   
    TH1* newphimod2_; 
    sfile->GetObject("1005;1",newphimod2_); 
    newphimod2_; 
    newphimod2_->SetLineColor(scolor); 
   
    TH1* phimod3_; 
    rfile->GetObject("1006;1",phimod3_); 
    phimod3_; 
    phimod3_->SetLineColor(rcolor); 
   
    TH1* newphimod3_; 
    sfile->GetObject("1006;1",newphimod3_); 
    newphimod3_; 
    newphimod3_->SetLineColor(scolor); 
   
    TH1* phimod4_; 
    rfile->GetObject("1007;1",phimod4_); 
    phimod4_; 
    phimod4_->SetLineColor(rcolor); 
   
    TH1* newphimod4_; 
    sfile->GetObject("1007;1",newphimod4_); 
    newphimod4_; 
    newphimod4_->SetLineColor(scolor); 
   
   
    Ecal->cd(1);  
    phimod1_->Draw();  
    newphimod1_->Draw("same"); 
    Ecal->cd(2);  
    phimod2_->Draw();  
    newphimod2_->Draw("same"); 
    Ecal->cd(3);  
    phimod3_->Draw();  
    newphimod3_->Draw("same"); 
    Ecal->cd(4);  
    phimod4_->Draw();  
    newphimod4_->Draw("same"); 
    Ecal->Print("ECAL_Barrel_modules_MB_compare.eps");  

  }  

 // ECAL preshower plots
 
 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,2);
   
   
   TH1* etazp_;
   rfile->GetObject("1011;1",etazp_);
   etazp_;
   etazp_->SetLineColor(rcolor);
   
   TH1* newetazp_;
   sfile->GetObject("1011;1",newetazp_);
   newetazp_;
   newetazp_->SetLineColor(scolor);
   
   TH1* phizp_;
   rfile->GetObject("1012;1",phizp_);
   phizp_;
   phizp_->SetLineColor(rcolor);
   
   TH1* newphizp_;
   sfile->GetObject("1012;1",newphizp_);
   newphizp_;
   newphizp_->SetLineColor(scolor);
   
   TH1* etazm_;
   rfile->GetObject("1013;1",etazm_);
   etazm_;
   etazm_->SetLineColor(rcolor);
   
   TH1* newetazm_;
   sfile->GetObject("1013;1",newetazm_);
   newetazm_;
   newetazm_->SetLineColor(scolor);
   
   TH1* phizm_;
   rfile->GetObject("1014;1",phizm_);
   phizm_;
   phizm_->SetLineColor(rcolor);
   
   TH1* newphizm_;
   sfile->GetObject("1014;1",newphizm_);
   newphizm_;
   newphizm_->SetLineColor(scolor);
   
   
   Ecal->cd(1); 
   etazp_->Draw(); 
   newetazp_->Draw("same");
   Ecal->cd(2); 
   phizp_->Draw(); 
   newphizp_->Draw("same");
   Ecal->cd(3); 
   etazm_->Draw(); 
   newetazm_->Draw("same");
   Ecal->cd(4); 
   phizm_->Draw(); 
   newphizm_->Draw("same");
   Ecal->Print("ECAL_Preshower_MB_compare.eps"); 

 } 

}

