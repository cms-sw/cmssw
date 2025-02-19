#include "TFile.h"
#include "TTree.h"
#include "TText.h"

void EcalSimHitsPlotCompare( TString currentfile   = "EcalSimHitsValidation_new.root",
                             TString referencefile = "EcalSimHitsValidation_old.root" )
{
  
  gROOT ->Reset();
  char*  rfilename = referencefile ;
  char*  sfilename = currentfile ;
  
  int rcolor = 2;      // new plots
  int scolor = 4;      // old plots
  
  delete gROOT->GetListOfFiles()->FindObject(rfilename);
  delete gROOT->GetListOfFiles()->FindObject(sfilename);
  
  TText* te = new TText();
  te->SetTextSize(0.1);
  TFile * rfile = new TFile(rfilename);
  TFile * sfile = new TFile(sfilename);
  
  rfile->cd("DQMData/EcalHitsV/EcalSimHitsValidation");
  gDirectory->ls();
  
  sfile->cd("DQMData/EcalHitsV/EcalSimHitsValidation");
  gDirectory->ls();
  
  Char_t histo[200];
  
  gStyle->SetOptStat("n");
  
  gROOT->ProcessLine(".x HistoCompare.C");
  HistoCompare * myPV = new HistoCompare();


  //////////////////////////////////////////////////////////////

  // Particle Gun 
  if (1) {
    TCanvas *Ecal = new TCanvas("Ecal","Ecal",800,1000);
    Ecal->Divide(1,3);
    
    TH1 *meGunEnergy_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Gun Momentum;1", meGunEnergy_);
    meGunEnergy_;
    meGunEnergy_->SetLineColor(rcolor);
    
    TH1 *meGunEta_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Gun Eta;1", meGunEta_);
    meGunEta_;
    meGunEta_->SetLineColor(rcolor);
    
    TH1 *meGunPhi_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Gun Phi;1", meGunPhi_);
    meGunPhi_; 
    meGunPhi_->SetLineColor(rcolor); 
    
    TH1 *newmeGunEnergy_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Gun Momentum;1", newmeGunEnergy_);
    newmeGunEnergy_;
    newmeGunEnergy_->SetLineColor(scolor);
    
    TH1 *newmeGunEta_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Gun Eta;1", newmeGunEta_);
    newmeGunEta_;
    newmeGunEta_->SetLineColor(scolor);
    
    TH1 *newmeGunPhi_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Gun Phi;1", newmeGunPhi_);
    newmeGunPhi_; 
    newmeGunPhi_->SetLineColor(scolor); 
    
    
    Ecal->cd(1); 
    if ( meGunEnergy_ && newmeGunEnergy_ ) {
      meGunEnergy_   ->Draw(); 
      newmeGunEnergy_->Draw("same"); 
      myPV->PVCompute(meGunEnergy_ , newmeGunEnergy_ , te);
    }
    Ecal->cd(2); 
    if ( meGunEta_ && newmeGunEta_ ) {
      meGunEta_   ->Draw(); 
      newmeGunEta_->Draw("same"); 
      myPV->PVCompute(meGunEta_ , newmeGunEta_ , te);
    }
    Ecal->cd(3); 
    if ( meGunPhi_ && newmeGunPhi_ ) {
      meGunPhi_   ->Draw(); 
      newmeGunPhi_->Draw("same"); 
      myPV->PVCompute(meGunPhi_ , newmeGunPhi_ , te);
    }
    Ecal->Print("ParticleGun_compare.eps"); 
  }

  // Relative energy fraction
  if (1) {
    TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
    Ecal->Divide(1,3);
    
    TH1 *meEBEnergyFraction_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Barrel fraction of energy;1",meEBEnergyFraction_);
    meEBEnergyFraction_;
    meEBEnergyFraction_->SetLineColor(rcolor);
    
    TH1 *meEEEnergyFraction_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Endcap fraction of energy;1",meEEEnergyFraction_);
    meEEEnergyFraction_;
    meEEEnergyFraction_->SetLineColor(rcolor);
    
    TH1 *meESEnergyFraction_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Preshower fraction of energy;1",meESEnergyFraction_);
    meESEnergyFraction_;
    meESEnergyFraction_->SetLineColor(rcolor);

    TH1 *newmeEBEnergyFraction_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Barrel fraction of energy;1",newmeEBEnergyFraction_);
    newmeEBEnergyFraction_;
    newmeEBEnergyFraction_->SetLineColor(scolor);
    
    TH1 *newmeEEEnergyFraction_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Endcap fraction of energy;1",newmeEEEnergyFraction_);
    newmeEEEnergyFraction_;
    newmeEEEnergyFraction_->SetLineColor(scolor);
    
    TH1 *newmeESEnergyFraction_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EcalSimHitsValidation Preshower fraction of energy;1",newmeESEnergyFraction_);
    newmeESEnergyFraction_;
    newmeESEnergyFraction_->SetLineColor(scolor);
    
    Ecal->cd(1); 
    if (meEBEnergyFraction_ && newmeEBEnergyFraction_) {
      meEBEnergyFraction_   ->Draw();
      newmeEBEnergyFraction_->Draw("same");
      myPV->PVCompute(meEBEnergyFraction_, newmeEBEnergyFraction_, te);
    }
    
    Ecal->cd(2); 
    if (meEEEnergyFraction_ && newmeEEEnergyFraction_) {
      meEEEnergyFraction   _->Draw();
      newmeEEEnergyFraction_->Draw("same");
      myPV->PVCompute(meEEEnergyFraction_, newmeEEEnergyFraction_, te);
    }
    
    Ecal->cd(3); 
    if (meESEnergyFraction_ && newmeESEnergyFraction_) {
      meESEnergyFraction_   ->Draw();
      newmeESEnergyFraction_->Draw("same");
      myPV->PVCompute(meESEnergyFraction_, newmeESEnergyFraction_, te);
    }
    
    Ecal->Print("RelativeEnergyFraction_compare.eps");
  }

  // Hits multiplicity
  if (1) {
    TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
    Ecal->Divide(1,3);
    
    TH1 * meEBHits_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB hits multiplicity;1",meEBHits_);
    meEBHits_;
    meEBHits_->SetLineColor(rcolor);
    
    TH1 * meEEzpHits_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE+ hits multiplicity;1",meEEzpHits_);
    meEEzpHits_;
    meEEzpHits_->SetLineColor(rcolor);

    TH1 * meEEzmHits_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE- hits multiplicity;1",meEEzmHits_);
    meEEzmHits_;
    meEEzmHits_->SetLineColor(rcolor);
    
    TH1 * newmeEBHits_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB hits multiplicity;1",newmeEBHits_);
    newmeEBHits_;
    newmeEBHits_->SetLineColor(scolor);
    
    TH1 * newmeEEzpHits_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE+ hits multiplicity;1",newmeEEzpHits_);
    newmeEEzpHits_;
    newmeEEzpHits_->SetLineColor(scolor);

    TH1 * newmeEEzmHits_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE- hits multiplicity;1",newmeEEzmHits_);
    newmeEEzmHits_;
    newmeEEzmHits_->SetLineColor(scolor);
    
    Ecal->cd(1); 
    if (meEBHits_ && newmeEBHits_) {
      meEBHits_   ->Draw();
      newmeEBHits_->Draw("same");
      myPV->PVCompute(meEBHits_, newmeEBHits_, te);
    }

    Ecal->cd(2); 
    if (meEEzpHits_ && newmeEEzpHits_) {
      meEEzpHits_   ->Draw();
      newmeEEzpHits_->Draw("same");
      myPV->PVCompute(meEEzpHits_, newmeEEzpHits_, te);
    }

   Ecal->cd(3); 
    if (meEEzmHits_ && newmeEEzmHits_) {
      meEEzmHits_   ->Draw();
      newmeEEzmHits_->Draw("same");
      myPV->PVCompute(meEEzmHits_, newmeEEzmHits_, te);
    }
    
    Ecal->Print("HitsNumber_compare.eps");
  }
  
  
  // Crystals multiplicity
  if (1) {
    TCanvas *Ecal = new TCanvas("Ecal","Ecal",800,1000);
    Ecal->Divide(1,3);
    
    TH1 *meEBCrystals_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB crystals multiplicity;1",meEBCrystals_);
    meEBCrystals_;
    meEBCrystals_->SetLineColor(rcolor);
    
    TH1 *meEEzpCrystals_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE+ crystals multiplicity;1",meEEzpCrystals_);
    meEEzpCrystals_;
    meEEzpCrystals_->SetLineColor(rcolor);
    
    TH1 *meEEzmCrystals_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE- crystals multiplicity;1",meEEzmCrystals_);
    meEEzmCrystals_;
    meEEzmCrystals_->SetLineColor(rcolor);
    
    TH1 *newmeEBCrystals_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB crystals multiplicity;1",newmeEBCrystals_);
    newmeEBCrystals_;
    newmeEBCrystals_->SetLineColor(scolor);
    
    TH1 *newmeEEzpCrystals_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE+ crystals multiplicity;1",newmeEEzpCrystals_);
    newmeEEzpCrystals_;
    newmeEEzpCrystals_->SetLineColor(scolor);
    
    TH1 *newmeEEzmCrystals_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE- crystals multiplicity;1",newmeEEzmCrystals_);
    newmeEEzmCrystals_;
    newmeEEzmCrystals_->SetLineColor(scolor);
  
    Ecal->cd(1); 
    if (meEBCrystals_ && newmeEBCrystals_) {
      meEBCrystals_   ->Draw();
      newmeEBCrystals_->Draw("same");
      myPV->PVCompute(meEBCrystals_, newmeEBCrystals_, te);
    }
    
    Ecal->cd(2); 
    if (meEEzpCrystals_ && newmeEEzpCrystals_ ) {
      meEEzpCrystals_   ->Draw();
      newmeEEzpCrystals_->Draw("same");
      myPV->PVCompute(meEEzpCrystals_, newmeEEzpCrystals_, te);
    }
    
    Ecal->cd(3); 
    if (meEEzmCrystals_ && newmeEEzmCrystals_) {
      gPad->SetLogy();
      meEEzmCrystals_   ->Draw();
      newmeEEzmCrystals_->Draw("same");
      myPV->PVCompute(meEEzmCrystals_, newmeEEzmCrystals_, te);
    }
    
    Ecal->Print("CrystalsNumber_compare.eps");
  }
  
  
  // Barrel occupancy
  if (1) {
    TCanvas *Ecal = new TCanvas("Ecal","Ecal",800,1000);
    Ecal->Divide(1,2);
    
    TH2 *meEBoccupancy_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB occupancy;1",meEBoccupancy_);
    meEBoccupancy_;
    
    TH2 *newmeEBoccupancy_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB occupancy;1",newmeEBoccupancy_);
    newmeEBoccupancy_;
    
    if (meEBoccupancy_ && newmeEBoccupancy_){ 
      Ecal->cd(1);
      meEBoccupancy_->Draw("colz");
      Ecal->cd(2);
      newmeEBoccupancy_->Draw("colz");
      myPV->PVCompute(meEBoccupancy_ , newmeEBoccupancy_ , te);
    }
    Ecal->Print("Barrel_Occupancy_compare.eps");
  }


  // Endcap occupancy
  if (1) {
    TCanvas *Ecal = new TCanvas("Ecal","Ecal",800,1000);
    Ecal->Divide(2,2);
    
    TH2 *meEEzpOccupancy_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE+ occupancy",meEEzpOccupancy_);
    meEEzpOccupancy_;

    TH2 *meEEzmOccupancy_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE- occupancy",meEEzmOccupancy_);
    meEEzmOccupancy_;
    
    TH2 *newmeEEzpOccupancy_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE+ occupancy",newmeEEzpOccupancy_);
    newmeEEzpOccupancy_;

    TH2 *newmeEEzmOccupancy_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE- occupancy",newmeEEzmOccupancy_);
    newmeEEzmOccupancy_;
    
    if (meEEzpOccupancy_ && newmeEEzpOccupancy_){ 
      Ecal->cd(1);
      meEEzpOccupancy_->Draw("colz");
      Ecal->cd(2);
      newmeEEzpOccupancy_->Draw("colz");
      myPV->PVCompute(meEEzpOccupancy_ , newmeEEzpOccupancy_ , te);
    }

    if (meEEzmOccupancy_ && newmeEEzmOccupancy_){ 
      Ecal->cd(3);
      meEEzmOccupancy_->Draw("colz");
      Ecal->cd(4);
      newmeEEzmOccupancy_->Draw("colz");
      myPV->PVCompute(meEEzmOccupancy_ , newmeEEzmOccupancy_ , te);
    }
    Ecal->Print("Endcap_Occupancy_compare.eps");
  }


  // longitudinal shower profile
  if (1) {
    TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
    Ecal->Divide(1,2);
    
    TProfile *meEBLongitudinalShower_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB longitudinal shower profile;1",meEBLongitudinalShower_);
    meEBLongitudinalShower_;
    meEBLongitudinalShower_->SetLineColor(rcolor);
    
    TProfile *meEELongitudinalShower_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE longitudinal shower profile;1",meEELongitudinalShower_);
    meEELongitudinalShower_;
    meEELongitudinalShower_->SetLineColor(rcolor);
    
    TProfile *newmeEBLongitudinalShower_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB longitudinal shower profile;1",newmeEBLongitudinalShower_);
    newmeEBLongitudinalShower_;
    newmeEBLongitudinalShower_->SetLineColor(scolor);
    
    TProfile *newmeEELongitudinalShower_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE longitudinal shower profile;1",newmeEELongitudinalShower_);
    newmeEELongitudinalShower_;
    newmeEELongitudinalShower_->SetLineColor(scolor);
    
    Ecal->cd(1);
    if ( meEBLongitudinalShower_ && newmeEBLongitudinalShower_ ) {
      meEBLongitudinalShower_   ->Draw();
      newmeEBLongitudinalShower_->Draw("same");
      myPV->PVCompute(meEBLongitudinalShower_, newmeEBLongitudinalShower_, te);
    }
    Ecal->cd(2);
    if ( meEELongitudinalShower_ && newmeEELongitudinalShower_ ) {
      meEELongitudinalShower_   ->Draw();
      newmeEELongitudinalShower_->Draw("same");
      myPV->PVCompute(meEELongitudinalShower_, newmeEELongitudinalShower_, te);
    }
    Ecal->Print("LongitudinalShowerProfile_compare.eps");
  }
   
  // hits energy spectrum
  if (1) {
    TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
    Ecal->Divide(1,3);
    
    TH1 *meEBhitEnergy_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB hits energy spectrum;1",meEBhitEnergy_);
    meEBhitEnergy_; 
    meEBhitEnergy_->SetLineColor(rcolor);
    
    TH1 *meEEzpHitEnergy_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE+ hits energy spectrum;1",meEEzpHitEnergy_);
    meEEzpHitEnergy_; 
    meEEzpHitEnergy_->SetLineColor(rcolor);

    TH1 *meEEzmHitEnergy_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE- hits energy spectrum;1",meEEzmHitEnergy_);
    meEEzmHitEnergy_; 
    meEEzmHitEnergy_->SetLineColor(rcolor);
    
    TH1 *newmeEBhitEnergy_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB hits energy spectrum;1",newmeEBhitEnergy_);
    newmeEBhitEnergy_; 
    newmeEBhitEnergy_->SetLineColor(scolor);
    
    TH1 *newmeEEzpHitEnergy_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE+ hits energy spectrum;1",newmeEEzpHitEnergy_);
    newmeEEzpHitEnergy_; 
    newmeEEzpHitEnergy_->SetLineColor(scolor);

    TH1 *newmeEEzmHitEnergy_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE- hits energy spectrum;1",newmeEEzmHitEnergy_);
    newmeEEzmHitEnergy_; 
    newmeEEzmHitEnergy_->SetLineColor(scolor);

    Ecal->cd(1);
    if ( meEBhitEnergy_ && newmeEBhitEnergy_){
      gPad->SetLogy();
      meEBhitEnergy_   ->Draw();
      newmeEBhitEnergy_->Draw("same");
      myPV->PVCompute(meEBhitEnergy_, newmeEBhitEnergy_, te);
    }

    Ecal->cd(2);
    if ( meEEzpHitEnergy_ && newmeEEzpHitEnergy_){
      gPad->SetLogy();
      meEEzpHitEnergy_   ->Draw();
      newmeEEzpHitEnergy_->Draw("same");
      myPV->PVCompute(meEEzpHitEnergy_, newmeEEzpHitEnergy_, te);
    }

    Ecal->cd(3);
    if ( meEEzmHitEnergy_ && newmeEEzmHitEnergy_){
      gPad->SetLogy();
      meEEzmHitEnergy_   ->Draw();
      newmeEEzmHitEnergy_->Draw("same");
      myPV->PVCompute(meEEzmHitEnergy_, newmeEEzmHitEnergy_, te);
    }
    Ecal->Print("HitsEnergySpectrum_compare.eps");
  }

  if (1) {
    TH1 *meEBe1_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E1;1",meEBe1_);
    meEBe1_; 
    meEBe1_->SetLineColor(rcolor);
    
    TH1 *meEBe4_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E4;1",meEBe4_);
    meEBe4_; 
    meEBe4_->SetLineColor(rcolor);
    
    TH1 *meEBe9_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E9;1",meEBe9_);
    meEBe9_; 
    meEBe9_->SetLineColor(rcolor);
    
    TH1 *meEBe16_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E16;1",meEBe16_);
    meEBe16_; 
    meEBe16_->SetLineColor(rcolor);
    
    TH1 *meEBe25_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E25;1",meEBe25_);
    meEBe25_; 
    meEBe25_->SetLineColor(rcolor);

    TH1 *newmeEBe1_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E1;1",newmeEBe1_);
    newmeEBe1_; 
    newmeEBe1_->SetLineColor(scolor);
    
    TH1 *newmeEBe4_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E4;1",newmeEBe4_);
    newmeEBe4_; 
    newmeEBe4_->SetLineColor(scolor);
    
    TH1 *newmeEBe9_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E9;1",newmeEBe9_);
    newmeEBe9_; 
    newmeEBe9_->SetLineColor(scolor);
    
    TH1 *newmeEBe16_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E16;1",newmeEBe16_);
    newmeEBe16_; 
    newmeEBe16_->SetLineColor(scolor);
    
    TH1 *newmeEBe25_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E25;1",newmeEBe25_);
    newmeEBe25_; 
    newmeEBe25_->SetLineColor(scolor);
    
    TCanvas *Ecal1 = new TCanvas("Ecal1","Ecal1",800,1000);
    if ( meEBe1_ && newmeEBe1_){ 
      gPad->SetLogy();
      meEBe1_   ->Draw();
      newmeEBe1_->Draw("same");
      myPV->PVCompute(meEBe1_, newmeEBe1_, te);
    }
    Ecal1->Print("EB1x1_compare.eps");
    
    TCanvas *Ecal4 = new TCanvas("Ecal4","Ecal4",800,1000);
    if ( meEBe4_ && newmeEBe4_){ 
      gPad->SetLogy();
      meEBe4_   ->Draw();
      newmeEBe4_->Draw("same");
      myPV->PVCompute(meEBe4_, newmeEBe4_, te);
    }
    Ecal4->Print("EB2x2_compare.eps");
    
    TCanvas *Ecal9 = new TCanvas("Ecal9","Ecal9",800,1000);
    if ( meEBe9_ && newmeEBe9_){ 
      gPad->SetLogy();
      meEBe9_   ->Draw();
      newmeEBe9_->Draw("same");
      myPV->PVCompute(meEBe9_, newmeEBe9_, te);
    }
    Ecal9->Print("EB3x3_compare.eps");

    TCanvas *Ecal16 = new TCanvas("Ecal16","Ecal16",800,1000);
    if ( meEBe16_ && newmeEBe16_){ 
      gPad->SetLogy();
      meEBe16  _ ->Draw();
      newmeEBe16_->Draw("same");
      myPV->PVCompute(meEBe16_, newmeEBe16_, te);
    }
    Ecal16->Print("EB4x4_compare.eps");
    
    TCanvas * Ecal25 = new TCanvas("Ecal25","Ecal25",800,1000);
    if ( meEBe25_ && newmeEBe25_){ 
      gPad->SetLogy();
      meEBe25_   ->Draw();
      newmeEBe25_->Draw("same");
      myPV->PVCompute(meEBe25_, newmeEBe25_, te);
    }
    Ecal25->Print("EB5x5_compare.eps");
  }

  
  if (1) {
    TH1 *meEEe1_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E1;1",meEEe1_);
    meEEe1_; 
    meEEe1_->SetLineColor(rcolor);
    
    TH1 *meEEe4_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E4;1",meEEe4_);
    meEEe4_; 
    meEEe4_->SetLineColor(rcolor);
    
    TH1 *meEEe9_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E9;1",meEEe9_);
    meEEe9_; 
    meEEe9_->SetLineColor(rcolor);
    
    TH1 *meEEe16_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E16;1",meEEe16_);
    meEEe16_; 
    meEEe16_->SetLineColor(rcolor);
    
    TH1 *meEEe25_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E25;1",meEEe25_);
    meEEe25_; 
    meEEe25_->SetLineColor(rcolor);

    TH1 *newmeEEe1_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E1;1",newmeEEe1_);
    newmeEEe1_; 
    newmeEEe1_->SetLineColor(scolor);
    
    TH1 *newmeEEe4_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E4;1",newmeEEe4_);
    newmeEEe4_; 
    newmeEEe4_->SetLineColor(scolor);
    
    TH1 *newmeEEe9_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E9;1",newmeEEe9_);
    newmeEEe9_; 
    newmeEEe9_->SetLineColor(scolor);
    
    TH1 *newmeEEe16_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E16;1",newmeEEe16_);
    newmeEEe16_; 
    newmeEEe16_->SetLineColor(scolor);
    
    TH1 *newmeEEe25_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E25;1",newmeEEe25_);
    newmeEEe25_; 
    newmeEEe25_->SetLineColor(scolor);

    TCanvas *Ecal1 = new TCanvas("Ecal1","Ecal1",800,1000);
    if ( meEEe1_ && newmeEEe1_){ 
      gPad->SetLogy();
      meEEe1_   ->Draw();
      newmeEEe1_->Draw("same");
      myPV->PVCompute(meEEe1_, newmeEEe1_, te);
    }
    Ecal1->Print("EE1x1_compare.eps");

    TCanvas *Ecal4 = new TCanvas("Ecal4","Ecal4",800,1000);
    if ( meEEe4_ && newmeEEe4_){ 
      gPad->SetLogy();
      meEEe4_   ->Draw();
      newmeEEe4_->Draw("same");
      myPV->PVCompute(meEEe4_, newmeEEe4_, te);
    }
    Ecal4->Print("EE2x2_compare.eps");
    
    TCanvas *Ecal9 = new TCanvas("Ecal9","Ecal9",800,1000);
    if ( meEEe9_ && newmeEEe9_){ 
      gPad->SetLogy();
      meEEe9_   ->Draw();
      newmeEEe9_->Draw("same");
      myPV->PVCompute(meEEe9_, newmeEEe9_, te);
    }
    Ecal9->Print("EE3x3_compare.eps");
    
    TCanvas *Ecal16 = new TCanvas("Ecal16","Ecal16",800,1000);
    if ( meEEe16_ && newmeEEe16_){ 
      gPad->SetLogy();
      meEEe16  _ ->Draw();
      newmeEEe16_->Draw("same");
      myPV->PVCompute(meEEe16_, newmeEEe16_, te);
    }
    Ecal16->Print("EE4x4_compare.eps");
    
    TCanvas *Ecal25 = new TCanvas("Ecal25","Ecal25",800,1000);
    if ( meEEe25_ && newmeEEe25_){ 
      gPad->SetLogy();
      meEEe25 _  ->Draw();
      newmeEEe25_->Draw("same");
      myPV->PVCompute(meEEe25_, newmeEEe25_, te);
    }
    Ecal25->Print("EE5x5_compare.eps");
  }


  if (1) {
    TH1 *meEBe1oe4_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E1oE4;1",meEBe1oe4_);
    meEBe1oe4_; 
    meEBe1oe4_->SetLineColor(rcolor);
    
    TH1 *meEBe4oe9_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E4oE9;1",meEBe4oe9_);
    meEBe4oe9_; 
    meEBe4oe9_->SetLineColor(rcolor);
    
    TH1 *meEBe9oe16_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E9oE16;1",meEBe9oe16_);
    meEBe9oe16_; 
    meEBe9oe16_->SetLineColor(rcolor);

    TH1 *meEBe1oe25_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E1oE25;1",meEBe1oe25_);
    meEBe1oe25_; 
    meEBe1oe25_->SetLineColor(rcolor);
    
    TH1 *meEBe9oe25_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E9oE25;1",meEBe9oe25_);
    meEBe9oe25_; 
    meEBe9oe25_->SetLineColor(rcolor);
    
    TH1 *meEBe16oe25_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E16oE25;1",meEBe16oe25_);
    meEBe16oe25_; 
    meEBe16oe25_->SetLineColor(rcolor);
   
    TH1 *newmeEBe1oe4_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E1oE4;1",newmeEBe1oe4_);
    newmeEBe1oe4_; 
    newmeEBe1oe4_->SetLineColor(scolor);
    
    TH1 *newmeEBe4oe9_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E4oE9;1",newmeEBe4oe9_);
    newmeEBe4oe9_; 
    newmeEBe4oe9_->SetLineColor(scolor);
    
    TH1 *newmeEBe9oe16_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E9oE16;1",newmeEBe9oe16_);
    newmeEBe9oe16_; 
    newmeEBe9oe16_->SetLineColor(scolor);

    TH1 *newmeEBe1oe25_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E1oE25;1",newmeEBe1oe25_);
    newmeEBe1oe25_; 
    newmeEBe1oe25_->SetLineColor(scolor);
    
    TH1 *newmeEBe9oe25_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E9oE25;1",newmeEBe9oe25_);
    newmeEBe9oe25_; 
    newmeEBe9oe25_->SetLineColor(scolor);
    
    TH1 *newmeEBe16oe25_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EB E16oE25;1",newmeEBe16oe25_);
    newmeEBe16oe25_; 
    newmeEBe16oe25_->SetLineColor(scolor);
  
    TH1 *meEEe1oe4_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E1oE4;1",meEEe1oe4_);
    meEEe1oe4_; 
    meEEe1oe4_->SetLineColor(rcolor);
    
    TH1 *meEEe4oe9_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E4oE9;1",meEEe4oe9_);
    meEEe4oe9_; 
    meEEe4oe9_->SetLineColor(rcolor);
    
    TH1 *meEEe9oe16_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E9oE16;1",meEEe9oe16_);
    meEEe9oe16_; 
    meEEe9oe16_->SetLineColor(rcolor);

    TH1 *meEEe1oe25_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E1oE25;1",meEEe1oe25_);
    meEEe1oe25_; 
    meEEe1oe25_->SetLineColor(rcolor);
    
    TH1 *meEEe9oe25_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E9oE25;1",meEEe9oe25_);
    meEEe9oe25_; 
    meEEe9oe25_->SetLineColor(rcolor);
    
    TH1 *meEEe16oe25_; 
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E16oE25;1",meEEe16oe25_);
    meEEe16oe25_; 
    meEEe16oe25_->SetLineColor(rcolor);
    
    TH1 *newmeEEe1oe4_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E1oE4;1",newmeEEe1oe4_);
    newmeEEe1oe4_; 
    newmeEEe1oe4_->SetLineColor(scolor);
    
    TH1 *newmeEEe4oe9_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E4oE9;1",newmeEEe4oe9_);
    newmeEEe4oe9_; 
    newmeEEe4oe9_->SetLineColor(scolor);
    
    TH1 *newmeEEe9oe16_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E9oE16;1",newmeEEe9oe16_);
    newmeEEe9oe16_; 
    newmeEEe9oe16_->SetLineColor(scolor);
    
    TH1 *newmeEEe1oe25_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E1oE25;1",newmeEEe1oe25_);
    newmeEEe1oe25_; 
    newmeEEe1oe25_->SetLineColor(scolor);
    
    TH1 *newmeEEe9oe25_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E9oE25;1",newmeEEe9oe25_);
    newmeEEe9oe25_; 
    newmeEEe9oe25_->SetLineColor(scolor);

    TH1 *newmeEEe16oe25_; 
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE E16oE25;1",newmeEEe16oe25_);
    newmeEEe16oe25_; 
    newmeEEe16oe25_->SetLineColor(scolor);
  

    TCanvas *EcalEB1o4 = new TCanvas("EcalEB1o4","EcalEB1o4",800,1000);
    if ( meEBe1oe4_ && newmeEBe1oe4_){ 
      meEBe1oe4_   ->Draw();
      newmeEBe1oe4_->Draw("same");
      myPV->PVCompute(meEBe1oe4_, newmeEBe1oe4_, te);
    }
    EcalEB1o4->Print("EB1o4_compare.eps");

    TCanvas *EcalEB4o9 = new TCanvas("EcalEB4o9","EcalEB4o9",800,1000);
    if ( meEBe4oe9_ && newmeEBe4oe9_){ 
      meEBe4oe9_   ->Draw();
      newmeEBe4oe9_->Draw("same");
      myPV->PVCompute(meEBe4oe9_, newmeEBe4oe9_, te);
    }
    EcalEB4o9->Print("EB4o9_compare.eps");

    TCanvas *EcalEB9o16 = new TCanvas("EcalEB9o16","EcalEB9o16",800,1000);
    if ( meEBe9oe16_ && newmeEBe9oe16_){ 
      meEBe9oe16_   ->Draw();
      newmeEBe9oe16_->Draw("same");
      myPV->PVCompute(meEBe9oe16_, newmeEBe9oe16_, te);
    }
    EcalEB9o16->Print("EB9o16_compare.eps");

    TCanvas *EcalEB16o25 = new TCanvas("EcalEB16o25","EcalEB16o25",800,1000);
    if ( meEBe16oe25_ && newmeEBe16oe25_){ 
      meEBe16oe25_   ->Draw();
      newmeEBe16oe25_->Draw("same");
      myPV->PVCompute(meEBe16oe25_, newmeEBe16oe25_, te);
    }
    EcalEB16o25->Print("EB16o25_compare.eps");
   
    TCanvas *EcalEB9o25 = new TCanvas("EcalEB9o25","EcalEB9o25",800,1000);
    if ( meEBe9oe25_ && newmeEBe9oe25_){ 
      meEBe9oe25_   ->Draw();
      newmeEBe9oe25_->Draw("same");
      myPV->PVCompute(meEBe9oe25_, newmeEBe9oe25_, te);
    }
    EcalEB9o25->Print("EB9o25_compare.eps");
    
    TCanvas *EcalEE1o4 = new TCanvas("EcalEE1o4","EcalEE1o4",800,1000);
    if ( meEEe1oe4_ && newmeEEe1oe4_){ 
      meEEe1oe4_   ->Draw();
      newmeEEe1oe4_->Draw("same");
      myPV->PVCompute(meEEe1oe4_, newmeEEe1oe4_, te);
    }
    EcalEE1o4->Print("EE1o4_compare.eps");
    
    TCanvas * EcalEE4o9 = new TCanvas("EcalEE4o9","EcalEE4o9",800,1000);
    if ( meEEe4oe9_ && newmeEEe4oe9_){ 
      meEEe4oe9_   ->Draw();
      newmeEEe4oe9_->Draw("same");
      myPV->PVCompute(meEEe4oe9_, newmeEEe4oe9_, te);
    }
    EcalEE4o9->Print("EE4o9_compare.eps");

    TCanvas * EcalEE9o16 = new TCanvas("EcalEE9o16","EcalEE9o16",800,1000);
    if ( meEEe9oe16_ && newmeEEe9oe16_){ 
      meEEe9oe16_   ->Draw();
      newmeEEe9oe16_->Draw("same");
      myPV->PVCompute(meEEe9oe16_, newmeEEe9oe16_, te);
    }
    EcalEE9o16->Print("EE9o16_compare.eps");
    
    TCanvas * EcalEE16o25 = new TCanvas("EcalEE16o25","EcalEE16o25",800,1000);
    if ( meEEe16oe25_ && newmeEEe16oe25_){ 
      meEEe16oe25_   ->Draw();
      newmeEEe16oe25_->Draw("same");
      myPV->PVCompute(meEEe16oe25_, newmeEEe16oe25_, te);
    }
    EcalEE16o25->Print("EE16o25_compare.eps");
    
    TCanvas * EcalEE9o25 = new TCanvas("EcalEE9o25","EcalEE9o25",800,1000);
    if ( meEEe9oe25_ && newmeEEe9oe25_){ 
      meEEe9oe25_   ->Draw();
      newmeEEe9oe25_->Draw("same");
      myPV->PVCompute(meEEe9oe25_, newmeEEe9oe25_, te);
    }
    EcalEE9o25->Print("EE9o25_compare.eps");
  }


        
  // preshower
  if (1) {
    TH1 *meESHits1zp_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits layer 1 multiplicity z+", meESHits1zp_ );
    meESHits1zp_;
    meESHits1zp_->SetLineColor(rcolor);
    
    TH1 *meESHits2zp_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits layer 2 multiplicity z+", meESHits2zp_ );
    meESHits2zp_;
    meESHits2zp_->SetLineColor(rcolor);
    
    TH1 *meESHits1zm_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits layer 1 multiplicity z-", meESHits1zm_ );
    meESHits1zm_;
    meESHits1zm_->SetLineColor(rcolor);
    
    TH1 *meESHits2zm_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits layer 2 multiplicity z-", meESHits2zm_ );
    meESHits2zm_;
    meESHits2zm_->SetLineColor(rcolor);
    
    TH1 *newmeESHits1zp_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits layer 1 multiplicity z+", newmeESHits1zp_ );
    newmeESHits1zp_;
    newmeESHits1zp_->SetLineColor(scolor);
    
    TH1 *newmeESHits2zp_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits layer 2 multiplicity z+", newmeESHits2zp_ );
    newmeESHits2zp_;
    newmeESHits2zp_->SetLineColor(scolor);
    
    TH1 *newmeESHits1zm_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits layer 1 multiplicity z-", newmeESHits1zm_ );
    newmeESHits1zm_;
    newmeESHits1zm_->SetLineColor(scolor);
    
    TH1 *newmeESHits2zm_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits layer 2 multiplicity z-", newmeESHits2zm_ );
    newmeESHits2zm_;
    newmeESHits2zm_->SetLineColor(scolor);
    
    TCanvas *ESHits1zp = new TCanvas("ESHits1zp","ESHits1zp",800,1000);
    if ( meESHits1zp_ && newmeESHits1zp_ ){
      gPad->SetLogy();
      meESHits1zp_   ->Draw();
      newmeESHits1zp_->Draw("same");
      myPV->PVCompute(meESHits1zp_, newmeESHits1zp_, te);
    }
    ESHits1zp->Print("ESHits1zp_compare.eps");
    
    TCanvas *ESHits2zp = new TCanvas("ESHits2zp","ESHits2zp",800,1000);
    if ( meESHits2zp_ && newmeESHits2zp_ ){
      gPad->SetLogy();
      meESHits2zp_   ->Draw();
      newmeESHits2zp_->Draw("same");
      myPV->PVCompute(meESHits2zp_, newmeESHits2zp_, te);
    }
    ESHits2zp->Print("ESHits2zp_compare.eps");
    
    TCanvas *ESHits1zm = new TCanvas("ESHits1zm","ESHits1zm",800,1000);
    if ( meESHits1zm_ && newmeESHits1zm_ ){
      gPad->SetLogy();
      meESHits1zm_   ->Draw();
      newmeESHits1zm_->Draw("same");
      myPV->PVCompute(meESHits1zm_, newmeESHits1zm_, te);
    }
    ESHits1zm->Print("ESHits1zm_compare.eps");
    
    TCanvas *ESHits2zm = new TCanvas("ESHits2zm","ESHits2zm",800,1000);
    if ( meESHits2zm_ && newmeESHits2zm_ ){
      gPad->SetLogy();
      meESHits2zm_   ->Draw();
      newmeESHits2zm_->Draw("same");
      myPV->PVCompute(meESHits2zm_, newmeESHits2zm_, te);
    }
    ESHits2zm->Print("ESHits2zm_compare.eps");
  }
  

  if(1){
    
    TH1 *meESEnergy1zp_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits energy layer 1 z+", meESEnergy1zp_);
    meESEnergy1zp_;
    meESEnergy1zp_->SetLineColor(rcolor);
    
    TH1 *meESEnergy2zp_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits energy layer 2 z+", meESEnergy2zp_);
    meESEnergy2zp_;
    meESEnergy2zp_->SetLineColor(rcolor);
    
    TH1 *meESEnergy1zm_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits energy layer 1 z-", meESEnergy1zm_);
    meESEnergy1zm_;
    meESEnergy1zm_->SetLineColor(rcolor);
    
    TH1 *meESEnergy2zm_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits energy layer 2 z-", meESEnergy2zm_);
    meESEnergy2zm_;
    meESEnergy2zm_->SetLineColor(rcolor);
    
    TH1 *newmeESEnergy1zp_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits energy layer 1 z+", newmeESEnergy1zp_);
    newmeESEnergy1zp_;
    newmeESEnergy1zp_->SetLineColor(scolor);
    
    TH1 *newmeESEnergy2zp_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits energy layer 2 z+", newmeESEnergy2zp_);
    newmeESEnergy2zp_;
    newmeESEnergy2zp_->SetLineColor(scolor);
    
    TH1 *newmeESEnergy1zm_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits energy layer 1 z-", newmeESEnergy1zm_);
    newmeESEnergy1zm_;
    newmeESEnergy1zm_->SetLineColor(scolor);
    
    TH1 *newmeESEnergy2zm_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES hits energy layer 2 z-", newmeESEnergy2zm_);
    newmeESEnergy2zm_;
    newmeESEnergy2zm_->SetLineColor(scolor);
    
   
    TCanvas *ESEnergy1zm = new TCanvas("ESEnergy1zm","ESEnergy1zm",800,1000);
    if ( meESEnergy1zm_ && newmeESEnergy1zm_ ){
      meESEnergy1zm_   ->Draw();
      newmeESEnergy1zm_->Draw("same");
      myPV->PVCompute(meESEnergy1zm_, newmeESEnergy1zm_, te);
    }
    ESEnergy1zm->Print("ESEnergy1zm_compare.eps");
    
    TCanvas *ESEnergy1zp = new TCanvas("ESEnergy1zp","ESEnergy1zp",800,1000);
    if ( meESEnergy1zp_ && newmeESEnergy1zp_ ){
      meESEnergy1zp_   ->Draw();
      newmeESEnergy1zp_->Draw("same");
      myPV->PVCompute(meESEnergy1zp_, newmeESEnergy1zp_, te);
    }
    ESEnergy1zp->Print("ESEnergy1zp_compare.eps");
    
    TCanvas *ESEnergy2zm = new TCanvas("ESEnergy2zm","ESEnergy2zm",800,1000);
    if ( meESEnergy2zm_ && newmeESEnergy2zm_ ){
      meESEnergy2zm_   ->Draw();
      newmeESEnergy2zm_->Draw("same");
      myPV->PVCompute(meESEnergy2zm_, newmeESEnergy2zm_, te);
    }
    ESEnergy2zm->Print("ESEnergy2zm_compare.eps");
    
    TCanvas *ESEnergy2zp = new TCanvas("ESEnergy2zp","ESEnergy2zp",800,1000);
    if ( meESEnergy2zp_ && newmeESEnergy2zp_ ){
      meESEnergy2zp_   ->Draw();
      newmeESEnergy2zp_->Draw("same");
      myPV->PVCompute(meESEnergy2zp_, newmeESEnergy2zp_, te);
    }
    ESEnergy2zp->Print("ESEnergy2zp_compare.eps");
  }

  if(1){
    TH1 *meE1alphaE2zp_;                         
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES E1+07E2 z+", meE1alphaE2zp_);
    meE1alphaE2zp_;
    meE1alphaE2zp_->SetLineColor(rcolor);
    
    TH1 *meE1alphaE2zm_;                         
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES E1+07E2 z-", meE1alphaE2zm_);
    meE1alphaE2zm_;
    meE1alphaE2zm_->SetLineColor(rcolor);
    
    TH1 *me2eszpOver1eszp_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES ene2oEne1 z+", me2eszpOver1eszp_);
    me2eszpOver1eszp_;
    me2eszpOver1eszp_->SetLineColor(rcolor);
    
    TH1 *me2eszmOver1eszm_;
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES ene2oEne1 z-", me2eszmOver1eszm_);
    me2eszmOver1eszm_;
    me2eszmOver1eszm_->SetLineColor(rcolor);

    TProfile *meEEoverESzp_;                         
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE vs ES z+", meEEoverESzp_);
    meEEoverESzp_;                         
    meEEoverESzp_->SetLineColor(rcolor);

    TProfile *meEEoverESzm_;                         
    rfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE vs ES z-", meEEoverESzm_);
    meEEoverESzm_;                         
    meEEoverESzm_->SetLineColor(rcolor);

    TH1 *newmeE1alphaE2zp_;                         
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES E1+07E2 z+", newmeE1alphaE2zp_);
    newmeE1alphaE2zp_;
    newmeE1alphaE2zp_->SetLineColor(scolor);
    
    TH1 *newmeE1alphaE2zm_;                         
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES E1+07E2 z-", newmeE1alphaE2zm_);
    newmeE1alphaE2zm_;
    newmeE1alphaE2zm_->SetLineColor(scolor);
    
    TH1 *newme2eszpOver1eszp_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES ene2oEne1 z+", newme2eszpOver1eszp_);
    newme2eszpOver1eszp_;
    newme2eszpOver1eszp_->SetLineColor(scolor);
    
    TH1 *newme2eszmOver1eszm_;
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/ES ene2oEne1 z-", newme2eszmOver1eszm_);
    newme2eszmOver1eszm_;
    newme2eszmOver1eszm_->SetLineColor(scolor);

    TProfile *newmeEEoverESzp_;                         
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE vs ES z+", newmeEEoverESzp_);
    newmeEEoverESzp_;                         
    newmeEEoverESzp_->SetLineColor(scolor);

    TProfile *newmeEEoverESzm_;                         
    sfile->GetObject("DQMData/EcalHitsV/EcalSimHitsValidation/EE vs ES z-", newmeEEoverESzm_);
    newmeEEoverESzm_;                         
    newmeEEoverESzm_->SetLineColor(scolor);

    TCanvas *cE1alphaE2zp = new TCanvas("cE1alphaE2zp","E1 alpha E2 z+",800, 1000);
    if ( meE1alphaE2zp_ && newmeE1alphaE2zp_){
      meE1alphaE2zp_  ->Draw();
      newmeE1alphaE2zp_->Draw("same");
      myPV->PVCompute(meE1alphaE2zp_, newmeE1alphaE2zp_, te);
    }
    cE1alphaE2zp->Print("E1alphaE2zp_compare.eps");     
    
    TCanvas *cE1alphaE2zm = new TCanvas("cE1alphaE2zm","E1 alpha E2 z-",800, 1000);
    if ( meE1alphaE2zm_ && newmeE1alphaE2zm_){
      meE1alphaE2zm_  ->Draw();
      newmeE1alphaE2zm_->Draw("same");
      myPV->PVCompute(meE1alphaE2zm_, newmeE1alphaE2zm_, te);
    }
    cE1alphaE2zm->Print("E1alphaE2zm_compare.eps");     

    TCanvas *c2eszpOver1eszp = new TCanvas("c2eszpOver1eszp","c2eszpOver1eszp",800, 1000);
    if ( me2eszpOver1eszp_ && newme2eszpOver1eszp_ ){
      me2eszpOver1eszp_    ->Draw();
      newme2eszpOver1eszp_ ->Draw("same");
      myPV->PVCompute(me2eszpOver1eszp_, newme2eszpOver1eszp_, te); 
    }
    c2eszpOver1eszp->Print("2eszpOver1eszp_compare.eps");     
    
    TCanvas *c2eszmOver1eszm = new TCanvas("c2eszmOver1eszm","c2eszmOver1eszm",800, 1000);
    if ( me2eszmOver1eszm_ && newme2eszmOver1eszm_ ){
      me2eszmOver1eszm_    ->Draw();
      newme2eszmOver1eszm_ ->Draw("same");
      myPV->PVCompute(me2eszmOver1eszm_, newme2eszmOver1eszm_, te); 
    }
    c2eszmOver1eszm->Print("2eszmOver1eszm_compare.eps");

    TCanvas *cEEoverESzp = new TCanvas("cEEoverESzp","EE vs ES z+",800, 1000);
    if ( meEEoverESzp_ && newmeEEoverESzp_){
      meEEoverESzp_  ->Draw();
      newmeEEoverESzp_->Draw("same");
      myPV->PVCompute(meEEoverESzp_, newmeEEoverESzp_, te);
    }
    cEEoverESzp->Print("EEoverESzp_compare.eps");          

    TCanvas *cEEoverESzm = new TCanvas("cEEoverESzm","EE vs ES z-",800, 1000);
    if ( meEEoverESzm_ && newmeEEoverESzm_){
      meEEoverESzm_  ->Draw();
      newmeEEoverESzm_->Draw("same");
      myPV->PVCompute(meEEoverESzm_, newmeEEoverESzm_, te);
    }
    cEEoverESzm->Print("EEoverESzm_compare.eps");          
  } 
}

