#include "TFile.h"
#include "TTree.h"
#include "TText.h"

void EcalMixingModulePlots( TString inputfile = "EcalDigisValidation.root" )
{

 gROOT ->Reset();
 char*  rfilename = inputfile;

 delete gROOT->GetListOfFiles()->FindObject(rfilename);

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);

 rfile->cd("DQMData/EcalDigisV/EcalDigiTask");
 gDirectory->ls();

 Char_t histo[200];

 gStyle->SetOptStat("nemruoi");

//////////////////////////////////////////////////////////////


 // maximum Digi over Sim ratio

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBDigiMixRatiogt100ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over sim signal ratio gt 100 ADC;1",meEBDigiMixRatiogt100ADC_);
   meEBDigiMixRatiogt100ADC_;

   TH1 * meEEDigiMixRatiogt100ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over sim signal ratio gt 100 ADC;1",meEEDigiMixRatiogt100ADC_);
   meEEDigiMixRatiogt100ADC_;

   Ecal->cd(1);
   gPad->SetLogy(0);
   if ( meEBDigiMixRatiogt100ADC_ ) meEBDigiMixRatiogt100ADC_->Draw();
   gPad->SetLogy(1);
   Ecal->cd(2);
   gPad->SetLogy(0);
   if ( meEEDigiMixRatiogt100ADC_ ) meEEDigiMixRatiogt100ADC_->Draw();
   gPad->SetLogy(1);
   Ecal->Print("MaxADC_over_SimSignal_Ratio_gt100ADC.eps");
 } 

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBDigiMixRatioOriggt50pc_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over sim signal ratio signal gt 50pc gun;1",meEBDigiMixRatioOriggt50pc_);
   meEBDigiMixRatioOriggt50pc_;

   TH1 * meEEDigiMixRatioOriggt40pc_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over sim signal ratio signal gt 40pc gun;1",meEEDigiMixRatioOriggt40pc_);
   meEEDigiMixRatioOriggt40pc_;

   Ecal->cd(1);
   gPad->SetLogy(0);
   if ( meEBDigiMixRatioOriggt50pc_ ) meEBDigiMixRatioOriggt50pc_->Draw();
   gPad->SetLogy(1);
   Ecal->cd(2);
   gPad->SetLogy(0);
   if ( meEEDigiMixRatioOriggt40pc_ ) meEEDigiMixRatioOriggt40pc_->Draw();
   gPad->SetLogy(1);
   Ecal->Print("MaxADC_over_SimSignal_Ratio_Origgt40pc.eps");
 } 

 // Bunch crossing distribution

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,3);

   TH1 * meEBbunchCrossing_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel bunch crossing;1",meEBbunchCrossing_);
   meEBbunchCrossing_;

   TH1 * meEEbunchCrossing_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap bunch crossing;1",meEEbunchCrossing_);
   meEEbunchCrossing_;

   TH1 * meESbunchCrossing_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower bunch crossing;1",meESbunchCrossing_);
   meESbunchCrossing_;
   
   Ecal->cd(1); 
   if ( meEBbunchCrossing_ ) meEBbunchCrossing_->Draw(); 
   Ecal->cd(2); 
   if ( meEEbunchCrossing_ ) meEEbunchCrossing_->Draw(); 
   Ecal->cd(3); 
   if ( meESbunchCrossing_ ) meESbunchCrossing_->Draw(); 
   Ecal->Print("BunchCrossing.eps");

 }

 // Global shapes and ratios

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,3);
   
   TProfile * meEBShape_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel shape digi;1", meEBShape_);
   meEBShape_;

   TH1 * meEBShapeRatio_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel shape digi ratio;1",  meEBShapeRatio_ );
   meEBShapeRatio_;

   TProfile * meEEShape_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap shape digi;1", meEEShape_);
   meEEShape_;

   TH1 * meEEShapeRatio_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap shape digi ratio;1",  meEEShapeRatio_ );
   meEEShapeRatio_;

   TProfile * meESShape_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower shape digi;1", meESShape_);
   meESShape_;

   TH1 * meESShapeRatio_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower shape digi ratio;1",  meESShapeRatio_ );
   meESShapeRatio_;

   Ecal->cd(1);
   if ( meEBShape_ ) meEBShape_->Draw();
   Ecal->cd(2);
   if ( meEBShapeRatio_ ) meEBShapeRatio_->Draw();
   Ecal->cd(3);
   if ( meEEShape_ ) meEEShape_->Draw();
   Ecal->cd(4);
   if ( meEEShapeRatio_ ) meEEShapeRatio_->Draw();
   Ecal->cd(5);
   if ( meESShape_ ) meESShape_->Draw();
   Ecal->cd(6);
   if ( meESShapeRatio_ ) meESShapeRatio_->Draw();
   Ecal->Print("MixingModule_shape_and_ratio.eps");

 }

 // Bunch by bunch shapes

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(3,7);

   TProfile * meEBBunchShape_[21];
   for ( Int_t  i=0 ; i<21; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel shape bunch crossing %02d;1", i-10 );
     rfile->GetObject(histo,meEBBunchShape_[i]);
     meEBBunchShape_[i];
   }
   for ( Int_t  i=0 ; i<21; i++ ) {
     Ecal->cd(i+1);
     if ( meEBBunchShape_[i] ) meEBBunchShape_[i]->Draw();
   }
   Ecal->Print("Barrel_bunch_by_bunch_shapes.eps");
  
 }

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(3,7);

   TProfile * meEEBunchShape_[21];
   for ( Int_t  i=0 ; i<21; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap shape bunch crossing %02d;1", i-10 );
     rfile->GetObject(histo,meEEBunchShape_[i]);
     meEEBunchShape_[i];
   }
   for ( Int_t  i=0 ; i<21; i++ ) {
     Ecal->cd(i+1);
     if ( meEEBunchShape_[i] ) meEEBunchShape_[i]->Draw();
   }
   Ecal->Print("Endcap_bunch_by_bunch_shapes.eps");
  
 }

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(3,7);

   TProfile * meESBunchShape_[21];
   for ( Int_t  i=0 ; i<21; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower shape bunch crossing %02d;1", i-10 );
     rfile->GetObject(histo,meESBunchShape_[i]);
     meESBunchShape_[i];
   }
   for ( Int_t  i=0 ; i<21; i++ ) {
     Ecal->cd(i+1);
     if ( meESBunchShape_[i] ) meESBunchShape_[i]->Draw();
   }
   Ecal->Print("Preshower_bunch_by_bunch_shapes.eps");
  
 }

}
