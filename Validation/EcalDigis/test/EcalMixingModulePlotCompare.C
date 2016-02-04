#include "TFile.h"
#include "TTree.h"
#include "TText.h"

void EcalMixingModulePlotCompare( TString currentfile = "EcalDigisValidation_new.root",
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

 // maximum Digi over Sim ratio

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBDigiMixRatiogt100ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over sim signal ratio gt 100 ADC;1",meEBDigiMixRatiogt100ADC_);
   meEBDigiMixRatiogt100ADC_;
   meEBDigiMixRatiogt100ADC_->SetLineColor(rcolor);

   TH1 * meEEDigiMixRatiogt100ADC_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over sim signal ratio gt 100 ADC;1",meEEDigiMixRatiogt100ADC_);
   meEEDigiMixRatiogt100ADC_;
   meEEDigiMixRatiogt100ADC_->SetLineColor(rcolor);

   TH1 * newmeEBDigiMixRatiogt100ADC_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over sim signal ratio gt 100 ADC;1",newmeEBDigiMixRatiogt100ADC_);
   newmeEBDigiMixRatiogt100ADC_;
   newmeEBDigiMixRatiogt100ADC_->SetLineColor(scolor);

   TH1 * newmeEEDigiMixRatiogt100ADC_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over sim signal ratio gt 100 ADC;1",newmeEEDigiMixRatiogt100ADC_);
   newmeEEDigiMixRatiogt100ADC_;
   newmeEEDigiMixRatiogt100ADC_->SetLineColor(scolor);

   Ecal->cd(1);
   gPad->SetLogy(0);
   if ( meEBDigiMixRatiogt100ADC_ && newmeEBDigiMixRatiogt100ADC_ ) {
     meEBDigiMixRatiogt100ADC_->Draw();
     newmeEBDigiMixRatiogt100ADC_->Draw("same");
     myPV->PVCompute(meEBDigiMixRatiogt100ADC_ , newmeEBDigiMixRatiogt100ADC_ , te );
   }
   gPad->SetLogy(1);
   Ecal->cd(2);
   gPad->SetLogy(0);
   if ( meEEDigiMixRatiogt100ADC_ && newmeEEDigiMixRatiogt100ADC_ ) {
     meEEDigiMixRatiogt100ADC_->Draw();
     newmeEEDigiMixRatiogt100ADC_->Draw("same");
     myPV->PVCompute(meEEDigiMixRatiogt100ADC_ , newmeEEDigiMixRatiogt100ADC_ , te );
   }
   gPad->SetLogy(1);
   Ecal->Print("MaxADC_over_SimSignal_Ratio_gt100ADC_compare.eps");
 } 

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,2);

   TH1 * meEBDigiMixRatioOriggt50pc_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over sim signal ratio signal gt 50pc gun;1",meEBDigiMixRatioOriggt50pc_);
   meEBDigiMixRatioOriggt50pc_;
   meEBDigiMixRatioOriggt50pc_->SetLineColor(rcolor);

   TH1 * meEEDigiMixRatioOriggt40pc_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over sim signal ratio signal gt 40pc gun;1",meEEDigiMixRatioOriggt40pc_);
   meEEDigiMixRatioOriggt40pc_;
   meEEDigiMixRatioOriggt40pc_->SetLineColor(rcolor);

   TH1 * newmeEBDigiMixRatioOriggt50pc_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel maximum Digi over sim signal ratio signal gt 50pc gun;1",newmeEBDigiMixRatioOriggt50pc_);
   newmeEBDigiMixRatioOriggt50pc_;
   newmeEBDigiMixRatioOriggt50pc_->SetLineColor(scolor);

   TH1 * newmeEEDigiMixRatioOriggt40pc_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap maximum Digi over sim signal ratio signal gt 40pc gun;1",newmeEEDigiMixRatioOriggt40pc_);
   newmeEEDigiMixRatioOriggt40pc_;
   newmeEEDigiMixRatioOriggt40pc_->SetLineColor(scolor);

   Ecal->cd(1);
   gPad->SetLogy(0);
   if ( meEBDigiMixRatioOriggt50pc_ && newmeEBDigiMixRatioOriggt50pc_ ) {
     meEBDigiMixRatioOriggt50pc_->Draw();
     newmeEBDigiMixRatioOriggt50pc_->Draw("same");
     myPV->PVCompute(meEBDigiMixRatioOriggt50pc_ , newmeEBDigiMixRatioOriggt50pc_ , te );
   }
   gPad->SetLogy(1);
   Ecal->cd(2);
   gPad->SetLogy(0);
   if ( meEEDigiMixRatioOriggt40pc_ && newmeEEDigiMixRatioOriggt40pc_ ) {
     meEEDigiMixRatioOriggt40pc_->Draw();
     newmeEEDigiMixRatioOriggt40pc_->Draw("same");
     myPV->PVCompute(meEEDigiMixRatioOriggt40pc_ , newmeEEDigiMixRatioOriggt40pc_ , te );
   }
   gPad->SetLogy(1);
   Ecal->Print("MaxADC_over_SimSignal_Ratio_Origgt50pc_compare.eps");
 } 

 // Bunch crossing distribution

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(1,3);

   TH1 * meEBbunchCrossing_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel bunch crossing;1",meEBbunchCrossing_);
   meEBbunchCrossing_;
   meEBbunchCrossing_->SetLineColor(rcolor);

   TH1 * meEEbunchCrossing_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap bunch crossing;1",meEEbunchCrossing_);
   meEEbunchCrossing_;
   meEEbunchCrossing_->SetLineColor(rcolor);

   TH1 * meESbunchCrossing_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower bunch crossing;1",meESbunchCrossing_);
   meESbunchCrossing_;
   meESbunchCrossing_->SetLineColor(rcolor);

   TH1 * newmeEBbunchCrossing_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel bunch crossing;1",newmeEBbunchCrossing_);
   newmeEBbunchCrossing_;
   newmeEBbunchCrossing_->SetLineColor(scolor);

   TH1 * newmeEEbunchCrossing_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap bunch crossing;1",newmeEEbunchCrossing_);
   newmeEEbunchCrossing_;
   newmeEEbunchCrossing_->SetLineColor(scolor);

   TH1 * newmeESbunchCrossing_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower bunch crossing;1",newmeESbunchCrossing_);
   newmeESbunchCrossing_;
   newmeESbunchCrossing_->SetLineColor(scolor);
   
   Ecal->cd(1); 
   if ( meEBbunchCrossing_ && newmeEBbunchCrossing_ ) {
     meEBbunchCrossing_->Draw(); 
     newmeEBbunchCrossing_->Draw("same"); 
     myPV->PVCompute( meEBbunchCrossing_ , newmeEBbunchCrossing_ , te );
   }
   Ecal->cd(2); 
   if ( meEEbunchCrossing_ && newmeEEbunchCrossing_ ) {
     meEEbunchCrossing_->Draw(); 
     newmeEEbunchCrossing_->Draw("same"); 
     myPV->PVCompute( meEEbunchCrossing_ , newmeEEbunchCrossing_ , te );
   }
   Ecal->cd(3); 
   if ( meESbunchCrossing_ && newmeESbunchCrossing_ ) {
     meESbunchCrossing_->Draw(); 
     newmeESbunchCrossing_->Draw("same"); 
     myPV->PVCompute( meESbunchCrossing_ , newmeESbunchCrossing_ , te );
   }
   Ecal->Print("BunchCrossing_compare.eps");

 }


 // Global shapes and ratios

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(2,3);
   
   TProfile * meEBShape_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel shape digi;1", meEBShape_);
   meEBShape_;
   meEBShape_->SetLineColor(rcolor);

   TH1 * meEBShapeRatio_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel shape digi ratio;1",  meEBShapeRatio_ );
   meEBShapeRatio_;
   meEBShapeRatio_->SetLineColor(rcolor);

   TProfile * meEEShape_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap shape digi;1", meEEShape_);
   meEEShape_;
   meEEShape_->SetLineColor(rcolor);

   TH1 * meEEShapeRatio_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap shape digi ratio;1",  meEEShapeRatio_ );
   meEEShapeRatio_;
   meEEShapeRatio_->SetLineColor(rcolor);

   TProfile * meESShape_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower shape digi;1", meESShape_);
   meESShape_;
   meESShape_->SetLineColor(rcolor);

   TH1 * meESShapeRatio_;
   rfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower shape digi ratio;1",  meESShapeRatio_ );
   meESShapeRatio_;
   meESShapeRatio_->SetLineColor(rcolor);
   
   TProfile * newmeEBShape_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel shape digi;1", newmeEBShape_);
   newmeEBShape_;
   newmeEBShape_->SetLineColor(scolor);

   TH1 * newmeEBShapeRatio_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel shape digi ratio;1",  newmeEBShapeRatio_ );
   newmeEBShapeRatio_;
   newmeEBShapeRatio_->SetLineColor(scolor);

   TProfile * newmeEEShape_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap shape digi;1", newmeEEShape_);
   newmeEEShape_;
   newmeEEShape_->SetLineColor(scolor);

   TH1 * newmeEEShapeRatio_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap shape digi ratio;1",  newmeEEShapeRatio_ );
   newmeEEShapeRatio_;
   newmeEEShapeRatio_->SetLineColor(scolor);

   TProfile * newmeESShape_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower shape digi;1", newmeESShape_);
   newmeESShape_;
   newmeESShape_->SetLineColor(scolor);

   TH1 * newmeESShapeRatio_;
   sfile->GetObject("DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower shape digi ratio;1",  newmeESShapeRatio_ );
   newmeESShapeRatio_;
   newmeESShapeRatio_->SetLineColor(scolor);

   Ecal->cd(1);
   if ( meEBShape_ && newmeEBShape_ ) {
     meEBShape_->Draw();
     newmeEBShape_->Draw("same");
     myPV->PVCompute(meEBShape_ , newmeEBShape_ , te );
   }
   Ecal->cd(2);
   if ( meEBShapeRatio_ && newmeEBShapeRatio_ ) {
     meEBShapeRatio_->Draw();
     newmeEBShapeRatio_->Draw("same");
     myPV->PVCompute(meEBShapeRatio_ , newmeEBShapeRatio_ , te );
   }
   Ecal->cd(3);
   if ( meEEShape_ && newmeEEShape_ ) {
     meEEShape_->Draw();
     newmeEEShape_->Draw("same");
     myPV->PVCompute(meEEShape_ , newmeEEShape_ , te );
   }
   Ecal->cd(4);
   if ( meEEShapeRatio_ && newmeEEShapeRatio_ ) {
     meEEShapeRatio_->Draw();
     newmeEEShapeRatio_->Draw("same");
     myPV->PVCompute(meEEShapeRatio_ , newmeEEShapeRatio_ , te );
   }
   Ecal->cd(5);
   if ( meESShape_ && newmeESShape_ ) {
     meESShape_->Draw();
     newmeESShape_->Draw("same");
     myPV->PVCompute(meESShape_ , newmeESShape_ , te );
   }
   Ecal->cd(6);
   if ( meESShapeRatio_ && newmeESShapeRatio_ ) {
     meESShapeRatio_->Draw();
     newmeESShapeRatio_->Draw("same");
     myPV->PVCompute(meESShapeRatio_ , newmeESShapeRatio_ , te );
   }
   Ecal->Print("MixingModule_shape_and_ratio_compare.eps");

 }

 // Bunch by bunch shapes

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(3,7);

   TProfile * meEBBunchShape_[21];
   TProfile * newmeEBBunchShape_[21];
   for ( Int_t  i=0 ; i<21; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Barrel shape bunch crossing %02d;1", i-10 );
     rfile->GetObject(histo,meEBBunchShape_[i]);
     meEBBunchShape_[i];
     meEBBunchShape_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEBBunchShape_[i]);
     newmeEBBunchShape_[i];
     newmeEBBunchShape_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<21; i++ ) {
     Ecal->cd(i+1);
     if ( meEBBunchShape_[i] && newmeEBBunchShape_[i] ) {
       meEBBunchShape_[i]->Draw();
       newmeEBBunchShape_[i]->Draw("same");
       myPV->PVCompute(meEBBunchShape_[i] , newmeEBBunchShape_[i] , te );
     }
   }
   Ecal->Print("Barrel_bunch_by_bunch_shapes_compare.eps");
  
 }

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(3,7);

   TProfile * meEEBunchShape_[21];
   TProfile * newmeEEBunchShape_[21];
   for ( Int_t  i=0 ; i<21; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Endcap shape bunch crossing %02d;1", i-10 );
     rfile->GetObject(histo,meEEBunchShape_[i]);
     meEEBunchShape_[i];
     meEEBunchShape_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeEEBunchShape_[i]);
     newmeEEBunchShape_[i];
     newmeEEBunchShape_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<21; i++ ) {
     Ecal->cd(i+1);
     if ( meEEBunchShape_[i] && newmeEEBunchShape_[i] ) {
       meEEBunchShape_[i]->Draw();
       newmeEEBunchShape_[i]->Draw("same");
       myPV->PVCompute(meEEBunchShape_[i] , newmeEEBunchShape_[i] , te );
     }
   }
   Ecal->Print("Endcap_bunch_by_bunch_shapes_compare.eps");
  
 }

 if (1) {
   TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
   Ecal->Divide(3,7);

   TProfile * meESBunchShape_[21];
   TProfile * newmeESBunchShape_[21];
   for ( Int_t  i=0 ; i<21; i++ ) {
     sprintf (histo, "DQMData/EcalDigisV/EcalDigiTask/EcalDigiTask Preshower shape bunch crossing %02d;1", i-10 );
     rfile->GetObject(histo,meESBunchShape_[i]);
     meESBunchShape_[i];
     meESBunchShape_[i]->SetLineColor(rcolor);
     sfile->GetObject(histo,newmeESBunchShape_[i]);
     newmeESBunchShape_[i];
     newmeESBunchShape_[i]->SetLineColor(scolor);
   }
   for ( Int_t  i=0 ; i<21; i++ ) {
     Ecal->cd(i+1);
     if ( meESBunchShape_[i] && newmeESBunchShape_[i] ) {
       meESBunchShape_[i]->Draw();
       newmeESBunchShape_[i]->Draw("same");
       myPV->PVCompute(meESBunchShape_[i] , newmeESBunchShape_[i] , te );
     }
   }
   Ecal->Print("Preshower_bunch_by_bunch_shapes_compare.eps");
  
 }

  
}

