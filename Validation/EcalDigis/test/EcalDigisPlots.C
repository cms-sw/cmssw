#include "TFile.h"
#include "TTree.h"
#include "TText.h"

void EcalDigisPlots()
{

 gROOT ->Reset();
 char*  rfilename = "EcalDigisValidation.root";

 delete gROOT->GetListOfFiles()->FindObject(rfilename);

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);

 rfile->cd("DQMData/EcalDigiTask");
 gDirectory->ls();

 Char_t histo[200];

 gStyle->SetOptStat("nemruoi");

//////////////////////////////////////////////////////////////

// Particle Gun
 
 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,3);
   
   TH1* meGunEnergy_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Gun Momentum;1",meGunEnergy_);
   meGunEnergy_;
   
   TH1* meGunEta_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Gun Eta;1",meGunEta_);
   meGunEta_;
   
   TH1* meGunPhi_; 
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Gun Phi;1",meGunPhi_);
   meGunPhi_; 
   
   
   Ecal->cd(1); 
   meGunEnergy_->Draw(); 
   Ecal->cd(2); 
   meGunEta_->Draw(); 
   Ecal->cd(3); 
   meGunPhi_->Draw(); 
   Ecal->Print("ParticleGun.eps"); 
 }

 // Barrel occupancy

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);

   TH2 * meEBDigiOccupancy_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Barrel occupancy;1",meEBDigiOccupancy_);
   meEBDigiOccupancy_;

   Ecal->cd(1);
   meEBDigiOccupancy_->Draw();
   Ecal->Print("Barrel_Occupancy.eps");
 }


 // Endcap occupancy

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH2 * meEEDigiOccupancyzp_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Endcap occupancy z+;1",meEEDigiOccupancyzp_);
   meEEDigiOccupancyzp_;

   TH2 * meEEDigiOccupancyzm_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Endcap occupancy z-;1",meEEDigiOccupancyzm_);
   meEEDigiOccupancyzm_;

   Ecal->cd(1);
   meEEDigiOccupancyzp_->Draw();
   Ecal->cd(2);
   meEEDigiOccupancyzm_->Draw();
   Ecal->Print("Endcap_Occupancy.eps");
 }


 // global pulse shapes

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TProfile * meEBDigiADCGlobal_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Barrel global pulse shape;1",meEBDigiADCGlobal_) ;
   meEBDigiADCGlobal_;

   TProfile * meEEDigiADCGlobal_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Endcap global pulse shape;1",meEEDigiADCGlobal_) ;
   meEEDigiADCGlobal_;

   Ecal->cd(1);
   meEBDigiADCGlobal_->Draw();
   Ecal->cd(2);
   meEEDigiADCGlobal_->Draw();
   Ecal->Print("Global_pulse_shape.eps");
 }

 // maximum Digi over Sim ratio

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBDigiSimRatio_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Barrel maximum Digi over Sim ratio;1",meEBDigiSimRatio_);
   meEBDigiSimRatio_;

   TH1 * meEEDigiSimRatio_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Endcap maximum Digi over Sim ratio;1",meEEDigiSimRatio_);
   meEEDigiSimRatio_;

   Ecal->cd(1);
   gPad->SetLogy(0);
   meEBDigiSimRatio_->Draw();
   gPad->SetLogy(1);
   Ecal->cd(2);
   gPad->SetLogy(0);
   meEEDigiSimRatio_->Draw();
   gPad->SetLogy(1);
   Ecal->Print("MaxADC_over_Sim_Ratio.eps");
 } 

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBDigiSimRatiogt10ADC_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Barrel maximum Digi over Sim ratio gt 10 ADC;1",meEBDigiSimRatiogt10ADC_);
   meEBDigiSimRatiogt10ADC_;

   TH1 * meEEDigiSimRatiogt10ADC_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Endcap maximum Digi over Sim ratio gt 10 ADC;1",meEEDigiSimRatiogt10ADC_);
   meEEDigiSimRatiogt10ADC_;

   Ecal->cd(1);
   gPad->SetLogy(0);
   meEBDigiSimRatiogt10ADC_->Draw();
   gPad->SetLogy(1);
   Ecal->cd(2);
   gPad->SetLogy(0);
   meEEDigiSimRatiogt10ADC_->Draw();
   gPad->SetLogy(1);
   Ecal->Print("MaxADC_over_Sim_Ratio_gt10ADC.eps");
 } 

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBDigiSimRatiogt100ADC_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Barrel maximum Digi over Sim ratio gt 100 ADC;1",meEBDigiSimRatiogt100ADC_);
   meEBDigiSimRatiogt100ADC_;

   TH1 * meEEDigiSimRatiogt100ADC_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Endcap maximum Digi over Sim ratio gt 100 ADC;1",meEEDigiSimRatiogt100ADC_);
   meEEDigiSimRatiogt100ADC_;

   Ecal->cd(1);
   gPad->SetLogy(0);
   meEBDigiSimRatiogt100ADC_->Draw();
   gPad->SetLogy(1);
   Ecal->cd(2);
   gPad->SetLogy(0);
   meEEDigiSimRatiogt100ADC_->Draw();
   gPad->SetLogy(1);
   Ecal->Print("MaxADC_over_Sim_Ratio_gt100ADC.eps");
 } 

 // Gain switch check

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBnADCafterSwitch_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Barrel ADC counts after gain switch;1", meEBnADCafterSwitch_) ;
   meEBnADCafterSwitch_;

   TH1 * meEEnADCafterSwitch_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Endcap ADC counts after gain switch;1", meEEnADCafterSwitch_) ;
   meEEnADCafterSwitch_;

   Ecal->cd(1);
   meEBnADCafterSwitch_->Draw();
   Ecal->cd(2);
   meEEnADCafterSwitch_->Draw();
   Ecal->Print("Counts_after_gain_switch.eps");
 }

 // pedestal for pre-sample

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBPedestal_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Barrel pedestal for pre-sample;1",meEBPedestal_);
   meEBPedestal_;

   TH1 * meEEPedestal_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Endcap pedestal for pre-sample;1",meEEPedestal_);
   meEEPedestal_;

   Ecal->cd(1);
   meEBPedestal_->Draw();
   Ecal->cd(2);
   meEEPedestal_->Draw();
   Ecal->Print("Presample_pedestal.eps");
 } 

 // maximum position


 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBMaximumgt100ADC_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Barrel maximum position gt 100 ADC;1",meEBMaximumgt100ADC_);
   meEBMaximumgt100ADC_;

   TH1 * meEEMaximumgt100ADC_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Endcap maximum position gt 100 ADC;1",meEEMaximumgt100ADC_);
   meEEMaximumgt100ADC_;

   Ecal->cd(1);
   meEBMaximumgt100ADC_->Draw();
   Ecal->cd(2);
   meEEMaximumgt100ADC_->Draw();
   Ecal->Print("Maximum_position_gt100ADC.eps");
 }

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBMaximumgt10ADC_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Barrel maximum position gt 10 ADC;1",meEBMaximumgt10ADC_);
   meEBMaximumgt10ADC_;

   TH1 * meEEMaximumgt10ADC_;
   rfile->GetObject("DQMData/EcalDigiTask/EcalDigiTask Endcap maximum position gt 10 ADC;1",meEEMaximumgt10ADC_);
   meEEMaximumgt10ADC_;

   Ecal->cd(1);
   meEBMaximumgt10ADC_->Draw();
   Ecal->cd(2);
   meEEMaximumgt10ADC_->Draw();
   Ecal->Print("Maximum_position_gt10ADC.eps");
 }

 // Preshower ADC counts

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,3);

   TH1 * meESDigiADC_[3];
   for ( Int_t  i=0 ; i<3; i++ ) {
     sprintf (histo, "DQMData/EcalDigiTask/EcalDigiTask Preshower ADC pulse %02d;1", i+1) ;
     rfile->GetObject(histo,meESDigiADC_[i]);
     meESDigiADC_[i];
   }
   for ( Int_t  i=0 ; i<3; i++ ) {
     Ecal->cd(i+1);
     meESDigiADC_[i]->Draw();
   }
   Ecal->Print("Preshower_ADC_counts.eps");
 }
 
 // Barrel analog pulse

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEBDigiADCAnalog_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigiTask/EcalDigiTask Barrel analog pulse %02d;1", i+1) ;
     rfile->GetObject(histo,meEBDigiADCAnalog_[i]);
     meEBDigiADCAnalog_[i];
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     meEBDigiADCAnalog_[i]->Draw();
     gPad->SetLogy(1);
   }
   Ecal->Print("Barrel_analog_ADC_counts.eps");
 }

 // Barrel ADC counts gain 1

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEBDigiADCg1_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigiTask/EcalDigiTask Barrel ADC pulse %02d Gain 1;1", i+1) ;
     rfile->GetObject(histo,meEBDigiADCg1_[i]);
     meEBDigiADCg1_[i];
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     meEBDigiADCg1_[i]->Draw();
     gPad->SetLogy(1);
   }
   Ecal->Print("Barrel_ADC_counts_gain1.eps");
 }

 // Barrel ADC counts gain 6

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEBDigiADCg6_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigiTask/EcalDigiTask Barrel ADC pulse %02d Gain 6;1", i+1) ;
     rfile->GetObject(histo,meEBDigiADCg6_[i]);
     meEBDigiADCg6_[i];
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     meEBDigiADCg6_[i]->Draw();
     gPad->SetLogy(1);
   }
   Ecal->Print("Barrel_ADC_counts_gain6.eps");
 }

 // Barrel ADC counts gain 12

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEBDigiADCg12_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigiTask/EcalDigiTask Barrel ADC pulse %02d Gain 12;1", i+1) ;
     rfile->GetObject(histo,meEBDigiADCg12_[i]);
     meEBDigiADCg12_[i];
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     meEBDigiADCg12_[i]->Draw();
     gPad->SetLogy(1);
   }
   Ecal->Print("Barrel_ADC_counts_gain12.eps");
 }

 // Barrel gain distributions

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEBDigiGain_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigiTask/EcalDigiTask Barrel gain pulse %02d;1", i+1) ;
     rfile->GetObject(histo,meEBDigiGain_[i]);
     meEBDigiGain_[i];
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     meEBDigiGain_[i]->Draw();
     gPad->SetLogy(1);
   }
   Ecal->Print("Barrel_ADC_gain.eps");
 }
 
 // Endcap analog pulse

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEEDigiADCAnalog_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigiTask/EcalDigiTask Endcap analog pulse %02d;1", i+1) ;
     rfile->GetObject(histo,meEEDigiADCAnalog_[i]);
     meEEDigiADCAnalog_[i];
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     meEEDigiADCAnalog_[i]->Draw();
     gPad->SetLogy(1);
   }
   Ecal->Print("Endcap_analog_ADC_counts.eps");
 }

 // Endcap ADC counts gain 1

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEEDigiADCg1_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigiTask/EcalDigiTask Endcap ADC pulse %02d Gain 1;1", i+1) ;
     rfile->GetObject(histo,meEEDigiADCg1_[i]);
     meEEDigiADCg1_[i];
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     meEEDigiADCg1_[i]->Draw();
     gPad->SetLogy(1);
   }
   Ecal->Print("Endcap_ADC_counts_gain1.eps");
 }

 // Endcap ADC counts gain 6

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEEDigiADCg6_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigiTask/EcalDigiTask Endcap ADC pulse %02d Gain 6;1", i+1) ;
     rfile->GetObject(histo,meEEDigiADCg6_[i]);
     meEEDigiADCg6_[i];
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     meEEDigiADCg6_[i]->Draw();
     gPad->SetLogy(1);
   }
   Ecal->Print("Endcap_ADC_counts_gain6.eps");
 }

 // Endcap ADC counts gain 12

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEEDigiADCg12_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigiTask/EcalDigiTask Endcap ADC pulse %02d Gain 12;1", i+1) ;
     rfile->GetObject(histo,meEEDigiADCg12_[i]);
     meEEDigiADCg12_[i];
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     meEEDigiADCg12_[i]->Draw();
     gPad->SetLogy(1);
   }
   Ecal->Print("Endcap_ADC_counts_gain12.eps");
 }

 // Endcap gain distributions

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,5);

   TH1 * meEEDigiGain_[10];
   for ( Int_t  i=0 ; i<10; i++ ) {
     sprintf (histo, "DQMData/EcalDigiTask/EcalDigiTask Endcap gain pulse %02d;1", i+1) ;
     rfile->GetObject(histo,meEEDigiGain_[i]);
     meEEDigiGain_[i];
   }
   for ( Int_t  i=0 ; i<10; i++ ) {
     Ecal->cd(i+1);
     gPad->SetLogy(0);
     meEEDigiGain_[i]->Draw();
     gPad->SetLogy(1);
   }
   Ecal->Print("Endcap_ADC_gain.eps");
 }

}
