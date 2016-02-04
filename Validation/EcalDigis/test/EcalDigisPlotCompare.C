#include "TFile.h"
#include "TTree.h"
#include "TText.h"

void EcalDigisPlotCompare( TString currentfile = "EcalDigisValidation_new.root",
                           TString referencefile = "EcalDigisValidation_old.root" )
{

 gROOT ->Reset();
 char*  rfilename = referencefile ;
 char*  sfilename = currentfile ;

 int rcolor = 2;
 int scolor = 4;

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename);

 TText* te = new TText();
 te->SetTextSize(0.1);
 TFile * rfile = new TFile(rfilename);
 TFile * sfile = new TFile(sfilename);

 rfile->cd("DQMData/EcalDigisV/EcalDigiTask");
 gDirectory->ls();

 sfile->cd("DQMData/EcalDigisV/EcalDigiTask");
 gDirectory->ls();

 Char_t histo[200];

 gStyle->SetOptStat("n");

 gROOT->ProcessLine(".x HistoCompare.C");
 HistoCompare * myPV = new HistoCompare();

//////////////////////////////////////////////////////////////

// Particle Gun
 
 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,3);
   
   TH1* meGunEnergy_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Gun Momentum;1",meGunEnergy_);
   meGunEnergy_;
   meGunEnergy_->SetLineColor(rcolor);
   
   TH1* meGunEta_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Gun Eta;1",meGunEta_);
   meGunEta_;
   meGunEta_->SetLineColor(rcolor);
   
   TH1* meGunPhi_; 
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Gun Phi;1",meGunPhi_);
   meGunPhi_; 
   meGunPhi_->SetLineColor(rcolor); 
   
   TH1* newmeGunEnergy_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Gun Momentum;1",newmeGunEnergy_);
   newmeGunEnergy_;
   newmeGunEnergy_->SetLineColor(scolor);
   
   TH1* newmeGunEta_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Gun Eta;1",newmeGunEta_);
   newmeGunEta_;
   newmeGunEta_->SetLineColor(scolor);
   
   TH1* newmeGunPhi_; 
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Gun Phi;1",newmeGunPhi_);
   newmeGunPhi_; 
   newmeGunPhi_->SetLineColor(scolor); 
   
   
   Ecal->cd(1); 
   if ( meGunEnergy_ && newmeGunEnergy_ ) {
     meGunEnergy_->Draw(); 
     newmeGunEnergy_->Draw("same"); 
     myPV->PVCompute( meGunEnergy_ , newmeGunEnergy_ , te );
   }
   Ecal->cd(2); 
   if ( meGunEta_ && newmeGunEta_ ) {
     meGunEta_->Draw(); 
     newmeGunEta_->Draw("same"); 
     myPV->PVCompute( meGunEta_ , newmeGunEta_ , te );
   }
   Ecal->cd(3); 
   if ( meGunPhi_ && newmeGunPhi_ ) {
     meGunPhi_->Draw(); 
     newmeGunPhi_->Draw("same"); 
     myPV->PVCompute( meGunPhi_ , newmeGunPhi_ , te );
   }
   Ecal->Print("ParticleGun_compare.eps"); 
 }

 // Barrel occupancy

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH2 * meEBDigiOccupancy_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel occupancy;1",meEBDigiOccupancy_);
   meEBDigiOccupancy_;
   meEBDigiOccupancy_->SetLineColor(rcolor);

   TH2 * newmeEBDigiOccupancy_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel occupancy;1",newmeEBDigiOccupancy_);
   newmeEBDigiOccupancy_;
   newmeEBDigiOccupancy_->SetLineColor(scolor);

   if ( meEBDigiOccupancy_ && newmeEBDigiOccupancy_ ) {
     Ecal->cd(1);
     meEBDigiOccupancy_->Draw("colz");
     Ecal->cd(2);
     newmeEBDigiOccupancy_->Draw("colz");
     myPV->PVCompute(meEBDigiOccupancy_ , newmeEBDigiOccupancy_ , te);
   }
   Ecal->Print("Barrel_Occupancy_compare.eps");
 }


 // Endcap occupancy

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,2);

   TH2 * meEEDigiOccupancyzp_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap occupancy z+;1",meEEDigiOccupancyzp_);
   meEEDigiOccupancyzp_;
   meEEDigiOccupancyzp_->SetLineColor(rcolor);

   TH2 * meEEDigiOccupancyzm_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap occupancy z-;1",meEEDigiOccupancyzm_);
   meEEDigiOccupancyzm_;
   meEEDigiOccupancyzm_->SetLineColor(rcolor);

   TH2 * newmeEEDigiOccupancyzp_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap occupancy z+;1",newmeEEDigiOccupancyzp_);
   newmeEEDigiOccupancyzp_;
   newmeEEDigiOccupancyzp_->SetLineColor(scolor);

   TH2 * newmeEEDigiOccupancyzm_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap occupancy z-;1",newmeEEDigiOccupancyzm_);
   newmeEEDigiOccupancyzm_;
   newmeEEDigiOccupancyzm_->SetLineColor(scolor);

   if ( meEEDigiOccupancyzp_ && newmeEEDigiOccupancyzp_ && meEEDigiOccupancyzm_ && newmeEEDigiOccupancyzm_ ) {
     Ecal->cd(1);
     meEEDigiOccupancyzp_->Draw("colz");
     Ecal->cd(3);
     newmeEEDigiOccupancyzp_->Draw("colz");
     myPV->PVCompute(meEEDigiOccupancyzp_ , newmeEEDigiOccupancyzp_ , te);
     Ecal->cd(2);
     meEEDigiOccupancyzm_->Draw("colz");
     Ecal->cd(4);
     newmeEEDigiOccupancyzm_->Draw("colz");
     myPV->PVCompute(meEEDigiOccupancyzm_ , newmeEEDigiOccupancyzm_ , te);
   }
   Ecal->Print("Endcap_Occupancy_compare.eps");
 }

 // Multiplicities

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,2);

   TH1 * meEBDigiMultiplicity_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel digis multiplicity;1",meEBDigiMultiplicity_);
   meEBDigiMultiplicity_;
   meEBDigiMultiplicity_->SetLineColor(rcolor);

   TH1 * meESDigiMultiplicity_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower digis multiplicity;1",meESDigiMultiplicity_);
   meESDigiMultiplicity_;
   meESDigiMultiplicity_->SetLineColor(rcolor);

   TH1 * meEEDigiMultiplicityzp_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap multiplicity z+;1",meEEDigiMultiplicityzp_);
   meEEDigiMultiplicityzp_;
   meEEDigiMultiplicityzp_->SetLineColor(rcolor);

   TH1 * meEEDigiMultiplicityzm_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap multiplicity z-;1",meEEDigiMultiplicityzm_);
   meEEDigiMultiplicityzm_;
   meEEDigiMultiplicityzm_->SetLineColor(rcolor);

   TH1 * newmeEBDigiMultiplicity_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel digis multiplicity;1",newmeEBDigiMultiplicity_);
   newmeEBDigiMultiplicity_;
   newmeEBDigiMultiplicity_->SetLineColor(scolor);

   TH1 * newmeESDigiMultiplicity_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower digis multiplicity;1",newmeESDigiMultiplicity_);
   newmeESDigiMultiplicity_;
   newmeESDigiMultiplicity_->SetLineColor(scolor);

   TH1 * newmeEEDigiMultiplicityzp_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap multiplicity z+;1",newmeEEDigiMultiplicityzp_);
   newmeEEDigiMultiplicityzp_;
   newmeEEDigiMultiplicityzp_->SetLineColor(scolor);

   TH1 * newmeEEDigiMultiplicityzm_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap multiplicity z-;1",newmeEEDigiMultiplicityzm_);
   newmeEEDigiMultiplicityzm_;
   newmeEEDigiMultiplicityzm_->SetLineColor(scolor);

   Ecal->cd(1);
   if ( meEBDigiMultiplicity_ && newmeEBDigiMultiplicity_ ) {
     gPad->SetLogx(0);
     meEBDigiMultiplicity_->Draw();
     newmeEBDigiMultiplicity_->Draw("same");
     myPV->PVCompute(meEBDigiMultiplicity_ , newmeEBDigiMultiplicity_ , te);
     gPad->SetLogx(1);
   }
   Ecal->cd(2);
   if ( meESDigiMultiplicity_ && newmeESDigiMultiplicity_ ) {
     gPad->SetLogx(0);
     meESDigiMultiplicity_->Draw();
     newmeESDigiMultiplicity_->Draw("same");
     myPV->PVCompute(meESDigiMultiplicity_ , newmeESDigiMultiplicity_ , te);
     gPad->SetLogx(1);
   }
   Ecal->cd(3);
   if ( meEEDigiMultiplicityzp_ && newmeEEDigiMultiplicityzp_ ) {
     gPad->SetLogx(0);
     meEEDigiMultiplicityzp_->Draw();
     newmeEEDigiMultiplicityzp_->Draw("same");
     myPV->PVCompute(meEEDigiMultiplicityzp_ , newmeEEDigiMultiplicityzp_ , te);
     gPad->SetLogx(1);
   }
   Ecal->cd(4);
   if ( meEEDigiMultiplicityzm_ && newmeEEDigiMultiplicityzm_ ) {
     gPad->SetLogx(0);
     meEEDigiMultiplicityzm_->Draw();
     newmeEEDigiMultiplicityzm_->Draw("same");
     myPV->PVCompute(meEEDigiMultiplicityzm_ , newmeEEDigiMultiplicityzm_ , te);
     gPad->SetLogx(1);
   }
   Ecal->Print("Multiplicity_compare.eps");
 }


 // global pulse shapes

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TProfile * meEBDigiADCGlobal_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel global pulse shape;1",meEBDigiADCGlobal_) ;
   meEBDigiADCGlobal_;
   meEBDigiADCGlobal_->SetLineColor(rcolor);

   TProfile * meEEDigiADCGlobal_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap global pulse shape;1",meEEDigiADCGlobal_) ;
   meEEDigiADCGlobal_;
   meEEDigiADCGlobal_->SetLineColor(rcolor);

   TProfile * newmeEBDigiADCGlobal_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel global pulse shape;1",newmeEBDigiADCGlobal_) ;
   newmeEBDigiADCGlobal_;
   newmeEBDigiADCGlobal_->SetLineColor(scolor);

   TProfile * newmeEEDigiADCGlobal_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap global pulse shape;1",newmeEEDigiADCGlobal_) ;
   newmeEEDigiADCGlobal_;
   newmeEEDigiADCGlobal_->SetLineColor(scolor);

   Ecal->cd(1);
   if ( meEBDigiADCGlobal_ && newmeEBDigiADCGlobal_ ) {
     meEBDigiADCGlobal_->Draw();
     newmeEBDigiADCGlobal_->Draw("same");
     myPV->PVCompute(meEBDigiADCGlobal_ , newmeEBDigiADCGlobal_ , te);
   }
   Ecal->cd(2);
   if ( meEEDigiADCGlobal_ && newmeEEDigiADCGlobal_ ) { 
     meEEDigiADCGlobal_->Draw();
     newmeEEDigiADCGlobal_->Draw("same");
     myPV->PVCompute(meEEDigiADCGlobal_ , newmeEEDigiADCGlobal_ , te);
   }
   Ecal->Print("Global_pulse_shape_compare.eps");
 }

 // maximum Digi over Sim ratio

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBDigiSimRatio_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over Sim ratio;1",meEBDigiSimRatio_);
   meEBDigiSimRatio_;
   meEBDigiSimRatio_->SetLineColor(rcolor);

   TH1 * meEEDigiSimRatio_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over Sim ratio;1",meEEDigiSimRatio_);
   meEEDigiSimRatio_;
   meEEDigiSimRatio_->SetLineColor(rcolor);

   TH1 * newmeEBDigiSimRatio_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over Sim ratio;1",newmeEBDigiSimRatio_);
   newmeEBDigiSimRatio_;
   newmeEBDigiSimRatio_->SetLineColor(scolor);

   TH1 * newmeEEDigiSimRatio_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over Sim ratio;1",newmeEEDigiSimRatio_);
   newmeEEDigiSimRatio_;
   newmeEEDigiSimRatio_->SetLineColor(scolor);

   Ecal->cd(1);
   gPad->SetLogy(0);
   if ( meEBDigiSimRatio_ && newmeEBDigiSimRatio_ ) {
     meEBDigiSimRatio_->Draw();
     newmeEBDigiSimRatio_->Draw("same");
     myPV->PVCompute(meEBDigiSimRatio_ , newmeEBDigiSimRatio_ , te);
   }
   gPad->SetLogy(1);
   Ecal->cd(2);
   gPad->SetLogy(0);
   if ( meEEDigiSimRatio_ && newmeEEDigiSimRatio_ ) {
     meEEDigiSimRatio_->Draw();
     newmeEEDigiSimRatio_->Draw("same");
     myPV->PVCompute(meEEDigiSimRatio_ , newmeEEDigiSimRatio_ , te);
   }
   gPad->SetLogy(1);
   Ecal->Print("MaxADC_over_Sim_Ratio_compare.eps");
 } 

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBDigiSimRatiogt10ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over Sim ratio gt 10 ADC;1",meEBDigiSimRatiogt10ADC_);
   meEBDigiSimRatiogt10ADC_;
   meEBDigiSimRatiogt10ADC_->SetLineColor(rcolor);

   TH1 * meEEDigiSimRatiogt20ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over Sim ratio gt 20 ADC;1",meEEDigiSimRatiogt20ADC_);
   meEEDigiSimRatiogt20ADC_;
   meEEDigiSimRatiogt20ADC_->SetLineColor(rcolor);

   TH1 * newmeEBDigiSimRatiogt10ADC_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over Sim ratio gt 10 ADC;1",newmeEBDigiSimRatiogt10ADC_);
   newmeEBDigiSimRatiogt10ADC_;
   newmeEBDigiSimRatiogt10ADC_->SetLineColor(scolor);

   TH1 * newmeEEDigiSimRatiogt20ADC_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over Sim ratio gt 20 ADC;1",newmeEEDigiSimRatiogt20ADC_);
   newmeEEDigiSimRatiogt20ADC_;
   newmeEEDigiSimRatiogt20ADC_->SetLineColor(scolor);

   Ecal->cd(1);
   gPad->SetLogy(0);
   if ( meEBDigiSimRatiogt10ADC_ && newmeEBDigiSimRatiogt10ADC_ ) {
     meEBDigiSimRatiogt10ADC_->Draw();
     newmeEBDigiSimRatiogt10ADC_->Draw("same");
     myPV->PVCompute(meEBDigiSimRatiogt10ADC_ , newmeEBDigiSimRatiogt10ADC_ , te);
   }
   gPad->SetLogy(1);
   Ecal->cd(2);
   gPad->SetLogy(0);
   if ( meEEDigiSimRatiogt20ADC_ && newmeEEDigiSimRatiogt20ADC_ ) {
     meEEDigiSimRatiogt20ADC_->Draw();
     newmeEEDigiSimRatiogt20ADC_->Draw("same");
     myPV->PVCompute(meEEDigiSimRatiogt20ADC_ , newmeEEDigiSimRatiogt20ADC_ , te);
   }
   gPad->SetLogy(1);
   Ecal->Print("MaxADC_over_Sim_Ratio_gt10ADC_compare.eps");
 } 

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBDigiSimRatiogt100ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over Sim ratio gt 100 ADC;1",meEBDigiSimRatiogt100ADC_);
   meEBDigiSimRatiogt100ADC_;
   meEBDigiSimRatiogt100ADC_->SetLineColor(rcolor);

   TH1 * meEEDigiSimRatiogt100ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over Sim ratio gt 100 ADC;1",meEEDigiSimRatiogt100ADC_);
   meEEDigiSimRatiogt100ADC_;
   meEEDigiSimRatiogt100ADC_->SetLineColor(rcolor);

   TH1 * newmeEBDigiSimRatiogt100ADC_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over Sim ratio gt 100 ADC;1",newmeEBDigiSimRatiogt100ADC_);
   newmeEBDigiSimRatiogt100ADC_;
   newmeEBDigiSimRatiogt100ADC_->SetLineColor(scolor);

   TH1 * newmeEEDigiSimRatiogt100ADC_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over Sim ratio gt 100 ADC;1",newmeEEDigiSimRatiogt100ADC_);
   newmeEEDigiSimRatiogt100ADC_;
   newmeEEDigiSimRatiogt100ADC_->SetLineColor(scolor);

   Ecal->cd(1);
   gPad->SetLogy(0);
   if ( meEBDigiSimRatiogt100ADC_ && newmeEBDigiSimRatiogt100ADC_ ) {
     meEBDigiSimRatiogt100ADC_->Draw();
     newmeEBDigiSimRatiogt100ADC_->Draw("same");
     myPV->PVCompute(meEBDigiSimRatiogt100ADC_ , newmeEBDigiSimRatiogt100ADC_ , te );
   }
   gPad->SetLogy(1);
   Ecal->cd(2);
   gPad->SetLogy(0);
   if ( meEEDigiSimRatiogt100ADC_ && newmeEEDigiSimRatiogt100ADC_ ) {
     meEEDigiSimRatiogt100ADC_->Draw();
     newmeEEDigiSimRatiogt100ADC_->Draw("same");
     myPV->PVCompute(meEEDigiSimRatiogt100ADC_ , newmeEEDigiSimRatiogt100ADC_ , te );
   }
   gPad->SetLogy(1);
   Ecal->Print("MaxADC_over_Sim_Ratio_gt100ADC_compare.eps");
 } 

 // Gain switch check

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBnADCafterSwitch_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel ADC counts after gain switch;1", meEBnADCafterSwitch_) ;
   meEBnADCafterSwitch_;
   meEBnADCafterSwitch_->SetLineColor(rcolor);

   TH1 * meEEnADCafterSwitch_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap ADC counts after gain switch;1", meEEnADCafterSwitch_) ;
   meEEnADCafterSwitch_;
   meEEnADCafterSwitch_->SetLineColor(rcolor);

   TH1 * newmeEBnADCafterSwitch_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel ADC counts after gain switch;1", newmeEBnADCafterSwitch_) ;
   newmeEBnADCafterSwitch_;
   newmeEBnADCafterSwitch_->SetLineColor(scolor);

   TH1 * newmeEEnADCafterSwitch_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap ADC counts after gain switch;1", newmeEEnADCafterSwitch_) ;
   newmeEEnADCafterSwitch_;
   newmeEEnADCafterSwitch_->SetLineColor(scolor);

   Ecal->cd(1);
   if ( meEBnADCafterSwitch_ && newmeEBnADCafterSwitch_ ) {
     meEBnADCafterSwitch_->Draw();
     newmeEBnADCafterSwitch_->Draw("same");
     myPV->PVCompute(meEBnADCafterSwitch_ , newmeEBnADCafterSwitch_ , te );
   }
   Ecal->cd(2);
   if ( meEEnADCafterSwitch_ && newmeEEnADCafterSwitch_ ) {
     meEEnADCafterSwitch_->Draw();
     newmeEEnADCafterSwitch_->Draw("same");
     myPV->PVCompute(meEEnADCafterSwitch_ , newmeEEnADCafterSwitch_ , te );
   }
   Ecal->Print("Counts_after_gain_switch_compare.eps");
 }

 // pedestal for pre-sample

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBPedestal_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel pedestal for pre-sample;1",meEBPedestal_);
   meEBPedestal_;
   meEBPedestal_->SetLineColor(rcolor);

   TH1 * meEEPedestal_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap pedestal for pre-sample;1",meEEPedestal_);
   meEEPedestal_;
   meEEPedestal_->SetLineColor(rcolor);

   TH1 * newmeEBPedestal_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel pedestal for pre-sample;1",newmeEBPedestal_);
   newmeEBPedestal_;
   newmeEBPedestal_->SetLineColor(scolor);

   TH1 * newmeEEPedestal_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap pedestal for pre-sample;1",newmeEEPedestal_);
   newmeEEPedestal_;
   newmeEEPedestal_->SetLineColor(scolor);

   Ecal->cd(1);
   if ( meEBPedestal_ && newmeEBPedestal_ ) {
     meEBPedestal_->Draw();
     newmeEBPedestal_->Draw("same");
     myPV->PVCompute(meEBPedestal_ , newmeEBPedestal_ , te );
   }
   Ecal->cd(2);
   if ( meEEPedestal_ && newmeEEPedestal_ ) {
     meEEPedestal_->Draw();
     newmeEEPedestal_->Draw("same");
     myPV->PVCompute(meEEPedestal_ , newmeEEPedestal_ , te );
   }
   Ecal->Print("Presample_pedestal_compare.eps");
 } 

 // maximum position


 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBMaximumgt100ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum position gt 100 ADC;1",meEBMaximumgt100ADC_);
   meEBMaximumgt100ADC_;
   meEBMaximumgt100ADC_->SetLineColor(rcolor);

   TH1 * meEEMaximumgt100ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum position gt 100 ADC;1",meEEMaximumgt100ADC_);
   meEEMaximumgt100ADC_;
   meEEMaximumgt100ADC_->SetLineColor(rcolor);

   TH1 * newmeEBMaximumgt100ADC_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum position gt 100 ADC;1",newmeEBMaximumgt100ADC_);
   newmeEBMaximumgt100ADC_;
   newmeEBMaximumgt100ADC_->SetLineColor(scolor);

   TH1 * newmeEEMaximumgt100ADC_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum position gt 100 ADC;1",newmeEEMaximumgt100ADC_);
   newmeEEMaximumgt100ADC_;
   newmeEEMaximumgt100ADC_->SetLineColor(scolor);

   Ecal->cd(1);
   if ( meEBMaximumgt100ADC_ && newmeEBMaximumgt100ADC_ ) { 
     meEBMaximumgt100ADC_->Draw();
     newmeEBMaximumgt100ADC_->Draw("same");
     myPV->PVCompute(meEBMaximumgt100ADC_ , newmeEBMaximumgt100ADC_ , te );
   }
   Ecal->cd(2);
   if ( meEEMaximumgt100ADC_ && newmeEEMaximumgt100ADC_ ) {
     meEEMaximumgt100ADC_->Draw();
     newmeEEMaximumgt100ADC_->Draw("same");
     myPV->PVCompute(meEEMaximumgt100ADC_ , newmeEEMaximumgt100ADC_ , te );
   }
   Ecal->Print("Maximum_position_gt100ADC_compare.eps");
 }

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBMaximumgt10ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum position gt 10 ADC;1",meEBMaximumgt10ADC_);
   meEBMaximumgt10ADC_;
   meEBMaximumgt10ADC_->SetLineColor(rcolor);

   TH1 * meEEMaximumgt20ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum position gt 20 ADC;1",meEEMaximumgt20ADC_);
   meEEMaximumgt20ADC_;
   meEEMaximumgt20ADC_->SetLineColor(rcolor);

   TH1 * newmeEBMaximumgt10ADC_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum position gt 10 ADC;1",newmeEBMaximumgt10ADC_);
   newmeEBMaximumgt10ADC_;
   newmeEBMaximumgt10ADC_->SetLineColor(scolor);

   TH1 * newmeEEMaximumgt20ADC_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum position gt 20 ADC;1",newmeEEMaximumgt20ADC_);
   newmeEEMaximumgt20ADC_;
   newmeEEMaximumgt20ADC_->SetLineColor(scolor);

   Ecal->cd(1);
   if ( meEBMaximumgt10ADC_ && newmeEBMaximumgt10ADC_ ) {
     meEBMaximumgt10ADC_->Draw();
     newmeEBMaximumgt10ADC_->Draw("same");
     myPV->PVCompute(meEBMaximumgt10ADC_ , newmeEBMaximumgt10ADC_ , te ); 
   }
   Ecal->cd(2);
   if ( meEEMaximumgt20ADC_ && newmeEEMaximumgt20ADC_ ) {
     meEEMaximumgt20ADC_->Draw();
     newmeEEMaximumgt20ADC_->Draw("same");
     myPV->PVCompute(meEEMaximumgt20ADC_ , newmeEEMaximumgt20ADC_ , te ); 
   }
   Ecal->Print("Maximum_position_gt10ADC_compare.eps");
 }

 // Preshower ADC counts

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,3);

   TH1 * meESDigiADC_[3];
   TH1 * newmeESDigiADC_[3];
   for ( Int_t  i=0 ; i<3; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower ADC pulse %02d;1", i+1) ;
     rfile->GetObject(histo,meESDigiADC_[i]);
     meESDigiADC_[i];
     meESDigiADC_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeESDigiADC_[i]);
     newmeESDigiADC_[i];
     newmeESDigiADC_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<3; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     if ( meESDigiADC_[i] && newmeESDigiADC_[i] ) { 
       meESDigiADC_[i]->Draw();
       newmeESDigiADC_[i]->Draw("same");
       myPV->PVCompute(meESDigiADC_[i] , newmeESDigiADC_[i] , te );
     }
     gPad->SetLogy(1);
   }
   Ecal->Print("Preshower_ADC_counts_compare.eps");
 }
 
 // Barrel analog pulse

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEBDigiADCAnalog_[10];
   TH1 * newmeEBDigiADCAnalog_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel analog pulse %02d;1", i+1) ;
     rfile->GetObject(histo,meEBDigiADCAnalog_[i]);
     meEBDigiADCAnalog_[i];
     meEBDigiADCAnalog_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEBDigiADCAnalog_[i]);
     newmeEBDigiADCAnalog_[i];
     newmeEBDigiADCAnalog_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     if ( meEBDigiADCAnalog_[i] && newmeEBDigiADCAnalog_[i] ) {
       meEBDigiADCAnalog_[i]->Draw();
       newmeEBDigiADCAnalog_[i]->Draw("same");
       myPV->PVCompute(meEBDigiADCAnalog_[i] , newmeEBDigiADCAnalog_[i] , te );
     }
     gPad->SetLogy(1);
   }
   Ecal->Print("Barrel_analog_ADC_counts_compare.eps");
 }

 // Barrel ADC counts gain 1

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEBDigiADCg1_[10];
   TH1 * newmeEBDigiADCg1_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel ADC pulse %02d Gain 1;1", i+1) ;
     rfile->GetObject(histo,meEBDigiADCg1_[i]);
     meEBDigiADCg1_[i];
     meEBDigiADCg1_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEBDigiADCg1_[i]);
     newmeEBDigiADCg1_[i];
     newmeEBDigiADCg1_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     if ( meEBDigiADCg1_[i] && newmeEBDigiADCg1_[i] ) {
       meEBDigiADCg1_[i]->Draw();
       newmeEBDigiADCg1_[i]->Draw("same");
       myPV->PVCompute(meEBDigiADCg1_[i] , newmeEBDigiADCg1_[i] , te );
     }
     gPad->SetLogy(1);
   }
   Ecal->Print("Barrel_ADC_counts_gain1_compare.eps");
 }

 // Barrel ADC counts gain 6

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEBDigiADCg6_[10];
   TH1 * newmeEBDigiADCg6_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel ADC pulse %02d Gain 6;1", i+1) ;
     rfile->GetObject(histo,meEBDigiADCg6_[i]);
     meEBDigiADCg6_[i];
     meEBDigiADCg6_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEBDigiADCg6_[i]);
     newmeEBDigiADCg6_[i];
     newmeEBDigiADCg6_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     if ( meEBDigiADCg6_[i] && newmeEBDigiADCg6_[i] ) {
       meEBDigiADCg6_[i]->Draw();
       newmeEBDigiADCg6_[i]->Draw("same");
       myPV->PVCompute(meEBDigiADCg6_[i] , newmeEBDigiADCg6_[i] , te );
     }
     gPad->SetLogy(1);
   }
   Ecal->Print("Barrel_ADC_counts_gain6_compare.eps");
 }

 // Barrel ADC counts gain 12

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEBDigiADCg12_[10];
   TH1 * newmeEBDigiADCg12_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel ADC pulse %02d Gain 12;1", i+1) ;
     rfile->GetObject(histo,meEBDigiADCg12_[i]);
     meEBDigiADCg12_[i];
     meEBDigiADCg12_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEBDigiADCg12_[i]);
     newmeEBDigiADCg12_[i];
     newmeEBDigiADCg12_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     if ( meEBDigiADCg12_[i] && newmeEBDigiADCg12_[i] ) {
       meEBDigiADCg12_[i]->Draw();
       newmeEBDigiADCg12_[i]->Draw("same");
       myPV->PVCompute(meEBDigiADCg12_[i] , newmeEBDigiADCg12_[i] , te );
     }
     gPad->SetLogy(1);
   }
   Ecal->Print("Barrel_ADC_counts_gain12_compare.eps");
 }

 // Barrel gain distributions

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEBDigiGain_[10];
   TH1 * newmeEBDigiGain_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel gain pulse %02d;1", i+1) ;
     rfile->GetObject(histo,meEBDigiGain_[i]);
     meEBDigiGain_[i];
     meEBDigiGain_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEBDigiGain_[i]);
     newmeEBDigiGain_[i];
     newmeEBDigiGain_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     if ( meEBDigiGain_[i] && newmeEBDigiGain_[i] ) {
       meEBDigiGain_[i]->Draw();
       newmeEBDigiGain_[i]->Draw("same");
       myPV->PVCompute(meEBDigiGain_[i] , newmeEBDigiGain_[i] , te );
     }
     gPad->SetLogy(1);
   }
   Ecal->Print("Barrel_ADC_gain_compare.eps");
 }
 
 // Endcap analog pulse

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEEDigiADCAnalog_[10];
   TH1 * newmeEEDigiADCAnalog_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap analog pulse %02d;1", i+1) ;
     rfile->GetObject(histo,meEEDigiADCAnalog_[i]);
     meEEDigiADCAnalog_[i];
     meEEDigiADCAnalog_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEEDigiADCAnalog_[i]);
     newmeEEDigiADCAnalog_[i];
     newmeEEDigiADCAnalog_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     if ( meEEDigiADCAnalog_[i] && newmeEEDigiADCAnalog_[i] ) {
       meEEDigiADCAnalog_[i]->Draw();
       newmeEEDigiADCAnalog_[i]->Draw("same");
       myPV->PVCompute(meEEDigiADCAnalog_[i] , newmeEEDigiADCAnalog_[i] , te );
     }
     gPad->SetLogy(1);
   }
   Ecal->Print("Endcap_analog_ADC_counts_compare.eps");
 }

 // Endcap ADC counts gain 1

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEEDigiADCg1_[10];
   TH1 * newmeEEDigiADCg1_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap ADC pulse %02d Gain 1;1", i+1) ;
     rfile->GetObject(histo,meEEDigiADCg1_[i]);
     meEEDigiADCg1_[i];
     meEEDigiADCg1_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEEDigiADCg1_[i]);
     newmeEEDigiADCg1_[i];
     newmeEEDigiADCg1_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     if ( meEEDigiADCg1_[i] && newmeEEDigiADCg1_[i] ) {
       meEEDigiADCg1_[i]->Draw();
       newmeEEDigiADCg1_[i]->Draw("same");
       myPV->PVCompute(meEEDigiADCg1_[i] , newmeEEDigiADCg1_[i] , te );
     }
     gPad->SetLogy(1);
   }
   Ecal->Print("Endcap_ADC_counts_gain1_compare.eps");
 }

 // Endcap ADC counts gain 6

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEEDigiADCg6_[10];
   TH1 * newmeEEDigiADCg6_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap ADC pulse %02d Gain 6;1", i+1) ;
     rfile->GetObject(histo,meEEDigiADCg6_[i]);
     meEEDigiADCg6_[i];
     meEEDigiADCg6_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEEDigiADCg6_[i]);
     newmeEEDigiADCg6_[i];
     newmeEEDigiADCg6_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     if ( meEEDigiADCg6_[i] && newmeEEDigiADCg6_[i] ) {
       meEEDigiADCg6_[i]->Draw();
       newmeEEDigiADCg6_[i]->Draw("same");
       myPV->PVCompute(meEEDigiADCg6_[i] , newmeEEDigiADCg6_[i] , te );
     }
     gPad->SetLogy(1);
   }
   Ecal->Print("Endcap_ADC_counts_gain6_compare.eps");
 }

 // Endcap ADC counts gain 12

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEEDigiADCg12_[10];
   TH1 * newmeEEDigiADCg12_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap ADC pulse %02d Gain 12;1", i+1) ;
     rfile->GetObject(histo,meEEDigiADCg12_[i]);
     meEEDigiADCg12_[i];
     meEEDigiADCg12_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEEDigiADCg12_[i]);
     newmeEEDigiADCg12_[i];
     newmeEEDigiADCg12_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     if ( meEEDigiADCg12_[i] && newmeEEDigiADCg12_[i] ) {
       meEEDigiADCg12_[i]->Draw();
       newmeEEDigiADCg12_[i]->Draw("same");
       myPV->PVCompute(meEEDigiADCg12_[i] , newmeEEDigiADCg12_[i] , te );
     }
     gPad->SetLogy(1);
   }
   Ecal->Print("Endcap_ADC_counts_gain12_compare.eps");
 }

 // Endcap gain distributions

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEEDigiGain_[10];
   TH1 * newmeEEDigiGain_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap gain pulse %02d;1", i+1) ;
     rfile->GetObject(histo,meEEDigiGain_[i]);
     meEEDigiGain_[i];
     meEEDigiGain_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEEDigiGain_[i]);
     newmeEEDigiGain_[i];
     newmeEEDigiGain_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     if ( meEEDigiGain_[i] && newmeEEDigiGain_[i] ) {
       meEEDigiGain_[i]->Draw();
       newmeEEDigiGain_[i]->Draw("same");
       myPV->PVCompute(meEEDigiGain_[i] , newmeEEDigiGain_[i] , te );
     }
     gPad->SetLogy(1);
   }
   Ecal->Print("Endcap_ADC_gain_compare.eps");
 }

}

