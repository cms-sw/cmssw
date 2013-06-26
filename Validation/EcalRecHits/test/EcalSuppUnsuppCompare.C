#include "TFile.h"
#include "TTree.h"
#include "TText.h"

void EcalSuppUnsuppCompare( TString currentfile   = "EcalRecHitsValidation_supp.root",
                            TString referencefile = "EcalRecHitsValidation_unsupp.root")
{

 gROOT ->Reset();
 char*  rfilename = referencefile;
 char*  sfilename = currentfile;

 int rcolor = 2;
 int scolor = 4;

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename);

 TText* te = new TText();
 te->SetTextSize(0.1);
 TFile * rfile = new TFile(rfilename);
 TFile * sfile = new TFile(sfilename);

 rfile->cd("DQMData/EcalRecHitsV/EcalRecHitsTask");
 gDirectory->ls();

 sfile->cd("DQMData/EcalRecHitsV/EcalRecHitsTask");
 gDirectory->ls();

 Char_t histo[200];

 gStyle->SetOptStat("n");

 gROOT->ProcessLine(".x HistoCompare.C");
 HistoCompare * myPV = new HistoCompare();

 //////////////////////////////////////////////////////////////
 
 // General class: sim/rec hit ratios
 if (1) 
   {
     TH1 *meEBRecHitSimHitRatio_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalRecHitsTask/EcalRecHitsTask Barrel RecSimHit Ratio;1",meEBRecHitSimHitRatio_);
     meEBRecHitSimHitRatio_;
     meEBRecHitSimHitRatio_->SetLineColor(rcolor);
     
     TH1 *meEERecHitSimHitRatio_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalRecHitsTask/EcalRecHitsTask Endcap RecSimHit Ratio;1",meEERecHitSimHitRatio_);
     meEERecHitSimHitRatio_;
     meEERecHitSimHitRatio_->SetLineColor(rcolor);
     
     TH1 *meESRecHitSimHitRatio_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalRecHitsTask/EcalRecHitsTask Preshower RecSimHit Ratio;1",meESRecHitSimHitRatio_);
     meESRecHitSimHitRatio_;
     meESRecHitSimHitRatio_->SetLineColor(rcolor);

     TH1 *newmeEBRecHitSimHitRatio_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalRecHitsTask/EcalRecHitsTask Barrel RecSimHit Ratio;1",newmeEBRecHitSimHitRatio_);
     newmeEBRecHitSimHitRatio_;
     newmeEBRecHitSimHitRatio_->SetLineColor(scolor);
     
     TH1 *newmeEERecHitSimHitRatio_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalRecHitsTask/EcalRecHitsTask Endcap RecSimHit Ratio;1",newmeEERecHitSimHitRatio_);
     newmeEERecHitSimHitRatio_;
     newmeEERecHitSimHitRatio_->SetLineColor(scolor);
     
     TH1 *newmeESRecHitSimHitRatio_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalRecHitsTask/EcalRecHitsTask Preshower RecSimHit Ratio;1",newmeESRecHitSimHitRatio_);
     newmeESRecHitSimHitRatio_;
     newmeESRecHitSimHitRatio_->SetLineColor(scolor);

     // --------------------------------
     TCanvas *BarrelSimRec = new TCanvas("BarrelSimRec","BarrelSimRec",800,800);
     if ( meEBRecHitSimHitRatio_ && newmeEBRecHitSimHitRatio_ ) {
       meEBRecHitSimHitRatio_->Draw();
       newmeEBRecHitSimHitRatio_->Draw("same");
       myPV->PVCompute( meEBRecHitSimHitRatio_ , newmeEBRecHitSimHitRatio_ , te );       
     }
     BarrelSimRec->Print("Barrel_SimRecHitsRatio_compare.eps");
     delete BarrelSimRec;


     TCanvas *EndcapSimRec = new TCanvas("EndcapSimRec","EndcapSimRec",800,800);
     if ( meEERecHitSimHitRatio_ && newmeEERecHitSimHitRatio_ ) {
       meEERecHitSimHitRatio_->Draw();
       newmeEERecHitSimHitRatio_->Draw("same");
       myPV->PVCompute( meEERecHitSimHitRatio_ , newmeEERecHitSimHitRatio_ , te );       
     }
     EndcapSimRec->Print("Endcap_SimRecHitsRatio_compare.eps");
     delete EndcapSimRec;


     TCanvas *PreshowerSimRec = new TCanvas("PreshowerSimRec","PreshowerSimRec",800,800);
     if ( meESRecHitSimHitRatio_ && newmeESRecHitSimHitRatio_ ) {
       meESRecHitSimHitRatio_->Draw();
       newmeESRecHitSimHitRatio_->Draw("same");
       myPV->PVCompute( meESRecHitSimHitRatio_ , newmeESRecHitSimHitRatio_ , te );       
     }
     PreshowerSimRec->Print("Preshower_SimRecHitsRatio_compare.eps");
     delete PreshowerSimRec;
   }



 // ---------------------------------------------------------------------------------
 cout << endl;
 cout << "Barrel validation" << endl;
 rfile->cd("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask");
 gDirectory->ls();
 sfile->cd("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask");
 gDirectory->ls();
 
 
 // Barrel validation
 if (1) 
   {    
     TH1 *meEBUncalibRecHitsAmplitude_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Amplitude;1",meEBUncalibRecHitsAmplitude_);
     meEBUncalibRecHitsAmplitude_;
     meEBUncalibRecHitsAmplitude_->SetLineColor(rcolor);
      
     TH1 *meEBUncalibRecHitsPedestal_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Pedestal;1",meEBUncalibRecHitsPedestal_);
     meEBUncalibRecHitsPedestal_;
     meEBUncalibRecHitsPedestal_->SetLineColor(rcolor);
     
     TH1 *meEBUncalibRecHitsJitter_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Jitter;1",meEBUncalibRecHitsJitter_);
     meEBUncalibRecHitsJitter_;
     meEBUncalibRecHitsJitter_->SetLineColor(rcolor);
     
     TH1 *meEBUncalibRecHitsChi2_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Chi2;1",meEBUncalibRecHitsChi2_);
     meEBUncalibRecHitsChi2_;
     meEBUncalibRecHitsChi2_->SetLineColor(rcolor);

     TH1 *meEBUncalibRecHitMaxSampleRatio_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB RecHit Max Sample Ratio;1",meEBUncalibRecHitMaxSampleRatio_);
     meEBUncalibRecHitMaxSampleRatio_;
     meEBUncalibRecHitMaxSampleRatio_->SetLineColor(rcolor);

     TH1 *meEBUncalibRecHitsAmplitudeGt100adc_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Amplitude gt 100 adc counts;1",meEBUncalibRecHitsAmplitudeGt100adc_);
     meEBUncalibRecHitsAmplitudeGt100adc_;
     meEBUncalibRecHitsAmplitudeGt100adc_->SetLineColor(rcolor);
      
     TH1 *meEBUncalibRecHitsPedestalGt100adc_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Pedestal gt 100 adc counts;1",meEBUncalibRecHitsPedestalGt100adc_);
     meEBUncalibRecHitsPedestalGt100adc_;
     meEBUncalibRecHitsPedestalGt100adc_->SetLineColor(rcolor);
     
     TH1 *meEBUncalibRecHitsJitterGt100adc_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Jitter gt 100 adc counts;1",meEBUncalibRecHitsJitterGt100adc_);
     meEBUncalibRecHitsJitterGt100adc_;
     meEBUncalibRecHitsJitterGt100adc_->SetLineColor(rcolor);
     
     TH1 *meEBUncalibRecHitsChi2Gt100adc_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Chi2 gt 100 adc counts;1",meEBUncalibRecHitsChi2Gt100adc_);
     meEBUncalibRecHitsChi2Gt100adc_;
     meEBUncalibRecHitsChi2Gt100adc_->SetLineColor(rcolor);

     TH1 *meEBUncalibRecHitMaxSampleRatioGt100adc_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB RecHit Max Sample Ratio gt 100 adc counts;1",meEBUncalibRecHitMaxSampleRatioGt100adc_);
     meEBUncalibRecHitMaxSampleRatioGt100adc_;
     meEBUncalibRecHitMaxSampleRatioGt100adc_->SetLineColor(rcolor);

     TH1 *newmeEBUncalibRecHitsAmplitude_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Amplitude;1",newmeEBUncalibRecHitsAmplitude_);
     newmeEBUncalibRecHitsAmplitude_;
     newmeEBUncalibRecHitsAmplitude_->SetLineColor(scolor);
      
     TH1 *newmeEBUncalibRecHitsPedestal_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Pedestal;1",newmeEBUncalibRecHitsPedestal_);
     newmeEBUncalibRecHitsPedestal_;
     newmeEBUncalibRecHitsPedestal_->SetLineColor(scolor);
     
     TH1 *newmeEBUncalibRecHitsJitter_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Jitter;1",newmeEBUncalibRecHitsJitter_);
     newmeEBUncalibRecHitsJitter_;
     newmeEBUncalibRecHitsJitter_->SetLineColor(scolor);
     
     TH1 *newmeEBUncalibRecHitsChi2_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Chi2;1",newmeEBUncalibRecHitsChi2_);
     newmeEBUncalibRecHitsChi2_;
     newmeEBUncalibRecHitsChi2_->SetLineColor(scolor);

     TH1 *newmeEBUncalibRecHitMaxSampleRatio_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB RecHit Max Sample Ratio;1",newmeEBUncalibRecHitMaxSampleRatio_);
     newmeEBUncalibRecHitMaxSampleRatio_;
     newmeEBUncalibRecHitMaxSampleRatio_->SetLineColor(scolor);

     TH1 *newmeEBUncalibRecHitsAmplitudeGt100adc_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Amplitude gt 100 adc counts;1",newmeEBUncalibRecHitsAmplitudeGt100adc_);
     newmeEBUncalibRecHitsAmplitudeGt100adc_;
     newmeEBUncalibRecHitsAmplitudeGt100adc_->SetLineColor(scolor);
      
     TH1 *newmeEBUncalibRecHitsPedestalGt100adc_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Pedestal gt 100 adc counts;1",newmeEBUncalibRecHitsPedestalGt100adc_);
     newmeEBUncalibRecHitsPedestalGt100adc_;
     newmeEBUncalibRecHitsPedestalGt100adc_->SetLineColor(scolor);
     
     TH1 *newmeEBUncalibRecHitsJitterGt100adc_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Jitter gt 100 adc counts;1",newmeEBUncalibRecHitsJitterGt100adc_);
     newmeEBUncalibRecHitsJitterGt100adc_;
     newmeEBUncalibRecHitsJitterGt100adc_->SetLineColor(scolor);
     
     TH1 *newmeEBUncalibRecHitsChi2Gt100adc_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB Chi2 gt 100 adc counts;1",newmeEBUncalibRecHitsChi2Gt100adc_);
     newmeEBUncalibRecHitsChi2Gt100adc_;
     newmeEBUncalibRecHitsChi2Gt100adc_->SetLineColor(scolor);

     TH1 *newmeEBUncalibRecHitMaxSampleRatioGt100adc_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalBarrelRecHitsTask/EB RecHit Max Sample Ratio gt 100 adc counts;1",newmeEBUncalibRecHitMaxSampleRatioGt100adc_);
     newmeEBUncalibRecHitMaxSampleRatioGt100adc_;
     newmeEBUncalibRecHitMaxSampleRatioGt100adc_->SetLineColor(scolor);
      

      // --------------------------------
      TCanvas *Barrel = new TCanvas("Barrel","Barrel",800,800);
      Barrel->Divide(4,2);
      if ( meEBUncalibRecHitsAmplitude_ && newmeEBUncalibRecHitsAmplitude_ )
	{
	  Barrel->cd(1); 
	  gPad->SetLogy();  
	  meEBUncalibRecHitsAmplitude_->Draw(); 

	  Barrel->cd(5); 
	  gPad->SetLogy();  
	  newmeEBUncalibRecHitsAmplitude_->Draw(); 
	  myPV->PVCompute( meEBUncalibRecHitsAmplitude_ , newmeEBUncalibRecHitsAmplitude_ , te );       
	}

      if ( meEBUncalibRecHitsPedestal_ && newmeEBUncalibRecHitsPedestal_ )
	{
	  Barrel->cd(2);  
	  meEBUncalibRecHitsPedestal_->Draw(); 

	  Barrel->cd(6);  
	  newmeEBUncalibRecHitsPedestal_->Draw();
	  myPV->PVCompute( meEBUncalibRecHitsPedestal_ , newmeEBUncalibRecHitsPedestal_ , te );        
	}

      if ( meEBUncalibRecHitsJitter_ && newmeEBUncalibRecHitsJitter_ )
	{
	  Barrel->cd(3);  
	  meEBUncalibRecHitsJitter_->Draw(); 

	  Barrel->cd(7);  
	  newmeEBUncalibRecHitsJitter_->Draw();
	  myPV->PVCompute( meEBUncalibRecHitsJitter_ , newmeEBUncalibRecHitsJitter_ , te );       
	}

      if ( meEBUncalibRecHitMaxSampleRatio_ && newmeEBUncalibRecHitMaxSampleRatio_ )
	{
	  Barrel->cd(4);  
	  meEBUncalibRecHitMaxSampleRatio_->Draw(); 

	  Barrel->cd(8);  
	  newmeEBUncalibRecHitMaxSampleRatio_->Draw(); 
	  myPV->PVCompute( meEBUncalibRecHitMaxSampleRatio_ , newmeEBUncalibRecHitMaxSampleRatio_ , te );       
	}

      Barrel->Print("Barrel_compare.eps");
      delete Barrel;
   

      TCanvas *BarrelGt100adc = new TCanvas("BarrelGt100adc","BarrelGt100adc",800,800);
      BarrelGt100adc->Divide(4,2);
      if ( meEBUncalibRecHitsAmplitudeGt100adc_ && newmeEBUncalibRecHitsAmplitudeGt100adc_ )
	{
	  BarrelGt100adc->cd(1); 
	  gPad->SetLogy();  
	  meEBUncalibRecHitsAmplitudeGt100adc_->Draw(); 

	  BarrelGt100adc->cd(5); 
	  gPad->SetLogy();  
	  newmeEBUncalibRecHitsAmplitudeGt100adc_->Draw(); 
	  myPV->PVCompute( meEBUncalibRecHitsAmplitudeGt100adc_ , newmeEBUncalibRecHitsAmplitudeGt100adc_ , te );       
	}

      if ( meEBUncalibRecHitsPedestalGt100adc_ && newmeEBUncalibRecHitsPedestalGt100adc_ )
	{
	  BarrelGt100adc->cd(2);  
	  meEBUncalibRecHitsPedestalGt100adc_->Draw(); 

	  BarrelGt100adc->cd(6);  
	  newmeEBUncalibRecHitsPedestalGt100adc_->Draw();
	  myPV->PVCompute( meEBUncalibRecHitsPedestalGt100adc_ , newmeEBUncalibRecHitsPedestalGt100adc_ , te );        
	}

      if ( meEBUncalibRecHitsJitterGt100adc_ && newmeEBUncalibRecHitsJitterGt100adc_ )
	{
	  BarrelGt100adc->cd(3);  
	  meEBUncalibRecHitsJitterGt100adc_->Draw(); 

	  BarrelGt100adc->cd(7);  
	  newmeEBUncalibRecHitsJitterGt100adc_->Draw();
	  myPV->PVCompute( meEBUncalibRecHitsJitterGt100adc_ , newmeEBUncalibRecHitsJitterGt100adc_ , te );       
	}

      if ( meEBUncalibRecHitMaxSampleRatioGt100adc_ && newmeEBUncalibRecHitMaxSampleRatioGt100adc_ )
	{
	  BarrelGt100adc->cd(4);  
	  meEBUncalibRecHitMaxSampleRatioGt100adc_->Draw(); 

	  BarrelGt100adc->cd(8);  
	  newmeEBUncalibRecHitMaxSampleRatioGt100adc_->Draw(); 
	  myPV->PVCompute( meEBUncalibRecHitMaxSampleRatioGt100adc_ , newmeEBUncalibRecHitMaxSampleRatioGt100adc_ , te );       
	}

      BarrelGt100adc->Print("BarrelGt100adc_compare.eps");
      delete BarrelGt100adc;
   }





 // -----------------------------------------------------------------
 cout << endl;
 cout << "Endcap validation" << endl;
 rfile->cd("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask");
 gDirectory->ls();
 sfile->cd("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask");
 gDirectory->ls();
 

 
 // Endcap validation
 if (1) 
   {    
     TH1 *meEEUncalibRecHitsAmplitude_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Amplitude;1",meEEUncalibRecHitsAmplitude_);
     meEEUncalibRecHitsAmplitude_;
     meEEUncalibRecHitsAmplitude_->SetLineColor(rcolor);
      
     TH1 *meEEUncalibRecHitsPedestal_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Pedestal;1",meEEUncalibRecHitsPedestal_);
     meEEUncalibRecHitsPedestal_;
     meEEUncalibRecHitsPedestal_->SetLineColor(rcolor);
     
     TH1 *meEEUncalibRecHitsJitter_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Jitter;1",meEEUncalibRecHitsJitter_);
     meEEUncalibRecHitsJitter_;
     meEEUncalibRecHitsJitter_->SetLineColor(rcolor);
     
     TH1 *meEEUncalibRecHitsChi2_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Chi2;1",meEEUncalibRecHitsChi2_);
     meEEUncalibRecHitsChi2_;
     meEEUncalibRecHitsChi2_->SetLineColor(rcolor);

     TH1 *meEEUncalibRecHitMaxSampleRatio_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE RecHit Max Sample Ratio;1",meEEUncalibRecHitMaxSampleRatio_);
     meEEUncalibRecHitMaxSampleRatio_;
     meEEUncalibRecHitMaxSampleRatio_->SetLineColor(rcolor);

     TH1 *meEEUncalibRecHitsAmplitudeGt60adc_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Amplitude gt 60 adc counts;1",meEEUncalibRecHitsAmplitudeGt60adc_);
     meEEUncalibRecHitsAmplitudeGt60adc_;
     meEEUncalibRecHitsAmplitudeGt60adc_->SetLineColor(rcolor);
      
     TH1 *meEEUncalibRecHitsPedestalGt60adc_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Pedestal gt 60 adc counts;1",meEEUncalibRecHitsPedestalGt60adc_);
     meEEUncalibRecHitsPedestalGt60adc_;
     meEEUncalibRecHitsPedestalGt60adc_->SetLineColor(rcolor);
     
     TH1 *meEEUncalibRecHitsJitterGt60adc_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Jitter gt 60 adc counts;1",meEEUncalibRecHitsJitterGt60adc_);
     meEEUncalibRecHitsJitterGt60adc_;
     meEEUncalibRecHitsJitterGt60adc_->SetLineColor(rcolor);
     
     TH1 *meEEUncalibRecHitsChi2Gt60adc_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Chi2 gt 60 adc counts;1",meEEUncalibRecHitsChi2Gt60adc_);
     meEEUncalibRecHitsChi2Gt60adc_;
     meEEUncalibRecHitsChi2Gt60adc_->SetLineColor(rcolor);

     TH1 *meEEUncalibRecHitMaxSampleRatioGt60adc_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE RecHit Max Sample Ratio gt 60 adc counts;1",meEEUncalibRecHitMaxSampleRatioGt60adc_);
     meEEUncalibRecHitMaxSampleRatioGt60adc_;
     meEEUncalibRecHitMaxSampleRatioGt60adc_->SetLineColor(rcolor);

     TH1 *newmeEEUncalibRecHitsAmplitude_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Amplitude;1",newmeEEUncalibRecHitsAmplitude_);
     newmeEEUncalibRecHitsAmplitude_;
     newmeEEUncalibRecHitsAmplitude_->SetLineColor(scolor);
      
     TH1 *newmeEEUncalibRecHitsPedestal_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Pedestal;1",newmeEEUncalibRecHitsPedestal_);
     newmeEEUncalibRecHitsPedestal_;
     newmeEEUncalibRecHitsPedestal_->SetLineColor(scolor);
     
     TH1 *newmeEEUncalibRecHitsJitter_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Jitter;1",newmeEEUncalibRecHitsJitter_);
     newmeEEUncalibRecHitsJitter_;
     newmeEEUncalibRecHitsJitter_->SetLineColor(scolor);
     
     TH1 *newmeEEUncalibRecHitsChi2_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Chi2;1",newmeEEUncalibRecHitsChi2_);
     newmeEEUncalibRecHitsChi2_;
     newmeEEUncalibRecHitsChi2_->SetLineColor(scolor);
      
     TH1 *newmeEEUncalibRecHitMaxSampleRatio_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE RecHit Max Sample Ratio;1",newmeEEUncalibRecHitMaxSampleRatio_);
     newmeEEUncalibRecHitMaxSampleRatio_;
     newmeEEUncalibRecHitMaxSampleRatio_->SetLineColor(scolor);

     TH1 *newmeEEUncalibRecHitsAmplitudeGt60adc_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Amplitude gt 60 adc counts;1",newmeEEUncalibRecHitsAmplitudeGt60adc_);
     newmeEEUncalibRecHitsAmplitudeGt60adc_;
     newmeEEUncalibRecHitsAmplitudeGt60adc_->SetLineColor(scolor);
      
     TH1 *newmeEEUncalibRecHitsPedestalGt60adc_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Pedestal gt 60 adc counts;1",newmeEEUncalibRecHitsPedestalGt60adc_);
     newmeEEUncalibRecHitsPedestalGt60adc_;
     newmeEEUncalibRecHitsPedestalGt60adc_->SetLineColor(scolor);
     
     TH1 *newmeEEUncalibRecHitsJitterGt60adc_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Jitter gt 60 adc counts;1",newmeEEUncalibRecHitsJitterGt60adc_);
     newmeEEUncalibRecHitsJitterGt60adc_;
     newmeEEUncalibRecHitsJitterGt60adc_->SetLineColor(scolor);
     
     TH1 *newmeEEUncalibRecHitsChi2Gt60adc_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE Chi2 gt 60 adc counts;1",newmeEEUncalibRecHitsChi2Gt60adc_);
     newmeEEUncalibRecHitsChi2Gt60adc_;
     newmeEEUncalibRecHitsChi2Gt60adc_->SetLineColor(scolor);
      
     TH1 *newmeEEUncalibRecHitMaxSampleRatioGt60adc_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalEndcapRecHitsTask/EE RecHit Max Sample Ratio gt 60 adc counts;1",newmeEEUncalibRecHitMaxSampleRatioGt60adc_);
     newmeEEUncalibRecHitMaxSampleRatioGt60adc_;
     newmeEEUncalibRecHitMaxSampleRatioGt60adc_->SetLineColor(scolor);


      // --------------------------------
      TCanvas *Endcap = new TCanvas("Endcap","Endcap",800,800);
      Endcap->Divide(3,2);
      if ( meEEUncalibRecHitsAmplitude_ && newmeEEUncalibRecHitsAmplitude_ )
	{
	  Endcap->cd(1); 
	  gPad->SetLogy();  
	  meEEUncalibRecHitsAmplitude_->Draw(); 

	  Endcap->cd(4); 
	  gPad->SetLogy();  
	  newmeEEUncalibRecHitsAmplitude_->Draw(); 
	  myPV->PVCompute( meEEUncalibRecHitsAmplitude_ , newmeEEUncalibRecHitsAmplitude_ , te );       
	}

      if ( meEEUncalibRecHitsPedestal_ && newmeEEUncalibRecHitsPedestal_ )
	{
	  Endcap->cd(2);  
	  meEEUncalibRecHitsPedestal_->Draw(); 

	  Endcap->cd(5);  
	  newmeEEUncalibRecHitsPedestal_->Draw();
	  myPV->PVCompute( meEEUncalibRecHitsPedestal_ , newmeEEUncalibRecHitsPedestal_ , te );        
	}

      if ( meEEUncalibRecHitsJitter_ && newmeEEUncalibRecHitsJitter_ )
	{
	  Endcap->cd(3);  
	  meEEUncalibRecHitsJitter_->Draw(); 

	  Endcap->cd(6);  
	  newmeEEUncalibRecHitsJitter_->Draw();
	  myPV->PVCompute( meEEUncalibRecHitsJitter_ , newmeEEUncalibRecHitsJitter_ , te );       
	}
      Endcap->Print("Endcap_compare.eps");
      delete Endcap;


      TCanvas *EndcapGt60adc = new TCanvas("EndcapGt60adc","EndcapGt60adc",800,800);
      EndcapGt60adc->Divide(3,2);
      if ( meEEUncalibRecHitsAmplitudeGt60adc_ && newmeEEUncalibRecHitsAmplitudeGt60adc_ )
	{
	  EndcapGt60adc->cd(1); 
	  gPad->SetLogy();  
	  meEEUncalibRecHitsAmplitudeGt60adc_->Draw(); 

	  EndcapGt60adc->cd(4); 
	  gPad->SetLogy();  
	  newmeEEUncalibRecHitsAmplitudeGt60adc_->Draw(); 
	  myPV->PVCompute( meEEUncalibRecHitsAmplitudeGt60adc_ , newmeEEUncalibRecHitsAmplitudeGt60adc_ , te );       
	}

      if ( meEEUncalibRecHitsPedestalGt60adc_ && newmeEEUncalibRecHitsPedestalGt60adc_ )
	{
	  EndcapGt60adc->cd(2);  
	  meEEUncalibRecHitsPedestalGt60adc_->Draw(); 
	  myPV->PVCompute( meEEUncalibRecHitsPedestalGt60adc_ , newmeEEUncalibRecHitsPedestalGt60adc_ , te );        

	  EndcapGt60adc->cd(5);  
	  newmeEEUncalibRecHitsPedestalGt60adc_->Draw();
	  myPV->PVCompute( meEEUncalibRecHitsPedestalGt60adc_ , newmeEEUncalibRecHitsPedestalGt60adc_ , te );        
	}

      if ( meEEUncalibRecHitsJitterGt60adc_ && newmeEEUncalibRecHitsJitterGt60adc_ )
	{
	  EndcapGt60adc->cd(3);  
	  meEEUncalibRecHitsJitterGt60adc_->Draw(); 

	  EndcapGt60adc->cd(6);  
	  newmeEEUncalibRecHitsJitterGt60adc_->Draw();
	  myPV->PVCompute( meEEUncalibRecHitsJitterGt60adc_ , newmeEEUncalibRecHitsJitterGt60adc_ , te );       
	}
      EndcapGt60adc->Print("EndcapGt60adc_compare.eps");
      delete EndcapGt60adc;
   }







 // ----------------------------------------------------------------
 cout << endl;
 cout << "Preshower validation" << endl;
 rfile->cd("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask");
 gDirectory->ls();
 sfile->cd("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask");
 gDirectory->ls();

  
 // Preshower validation
 if (1) 
   {    
     TH1 *meESRecHitsEnergy_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/ES Energy;1",meESRecHitsEnergy_);
     meESRecHitsEnergy_;
     meESRecHitsEnergy_->SetLineColor(rcolor);
     
     TH1 *meESRecHitsEnergy_zp1st_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/ES Energy Plane1 Side+;1",meESRecHitsEnergy_zp1st_);
     meESRecHitsEnergy_zp1st_;
     meESRecHitsEnergy_zp1st_->SetLineColor(rcolor);

     TH1 *meESRecHitsEnergy_zp2nd_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/ES Energy Plane2 Side+;1",meESRecHitsEnergy_zp2nd_);
     meESRecHitsEnergy_zp2nd_;
     meESRecHitsEnergy_zp2nd_->SetLineColor(rcolor);

     TH1 *meESRecHitsEnergy_zm1st_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/ES Energy Plane1 Side-;1",meESRecHitsEnergy_zm1st_);
     meESRecHitsEnergy_zm1st_;
     meESRecHitsEnergy_zm1st_->SetLineColor(rcolor);

     TH1 *meESRecHitsEnergy_zm2nd_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/ES Energy Plane2 Side-;1",meESRecHitsEnergy_zm2nd_);
     meESRecHitsEnergy_zm2nd_;
     meESRecHitsEnergy_zm2nd_->SetLineColor(rcolor);

     TH1 *meESEERecHitsEnergy_zp_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/Preshower EE vs ES energy Side+;1",meESEERecHitsEnergy_zp_);
     meESEERecHitsEnergy_zp_;
  
     TH1 *meESEERecHitsEnergy_zm_;
     rfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/Preshower EE vs ES energy Side-;1",meESEERecHitsEnergy_zm_);
     meESEERecHitsEnergy_zm_;

     TH1 *newmeESRecHitsEnergy_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/ES Energy;1",newmeESRecHitsEnergy_);
     newmeESRecHitsEnergy_;
     newmeESRecHitsEnergy_->SetLineColor(scolor);
     
     TH1 *newmeESRecHitsEnergy_zp1st_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/ES Energy Plane1 Side+;1",newmeESRecHitsEnergy_zp1st_);
     newmeESRecHitsEnergy_zp1st_;
     newmeESRecHitsEnergy_zp1st_->SetLineColor(scolor);

     TH1 *newmeESRecHitsEnergy_zp2nd_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/ES Energy Plane2 Side+;1",newmeESRecHitsEnergy_zp2nd_);
     newmeESRecHitsEnergy_zp2nd_;
     newmeESRecHitsEnergy_zp2nd_->SetLineColor(scolor);

     TH1 *newmeESRecHitsEnergy_zm1st_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/ES Energy Plane1 Side-;1",newmeESRecHitsEnergy_zm1st_);
     newmeESRecHitsEnergy_zm1st_;
     newmeESRecHitsEnergy_zm1st_->SetLineColor(scolor);

     TH1 *newmeESRecHitsEnergy_zm2nd_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/ES Energy Plane2 Side-;1",newmeESRecHitsEnergy_zm2nd_);
     newmeESRecHitsEnergy_zm2nd_;
     newmeESRecHitsEnergy_zm2nd_->SetLineColor(scolor);

     TH1 *newmeESEERecHitsEnergy_zp_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/Preshower EE vs ES energy Side+;1",newmeESEERecHitsEnergy_zp_);
     newmeESEERecHitsEnergy_zp_;
  
     TH1 *newmeESEERecHitsEnergy_zm_;
     sfile->GetObject("DQMData/EcalRecHitsV/EcalPreshowerRecHitsTask/Preshower EE vs ES energy Side-;1",newmeESEERecHitsEnergy_zm_);
     newmeESEERecHitsEnergy_zm_;

     
     
     // ----------------------------------------------
     TCanvas *ESEnergy = new TCanvas("ESEnergy","ESEnergy",800,800);
     ESEnergy->Divide(2,2);
     if ( meESRecHitsEnergy_zp1st_ && newmeESRecHitsEnergy_zp1st_ )
       {
	 ESEnergy->cd(1); gPad->SetLogy(); 
	 meESRecHitsEnergy_zp1st   _->Draw(); 
	 newmeESRecHitsEnergy_zp1st_->Draw("same"); 
       }
     if ( meESRecHitsEnergy_zp2nd_ && newmeESRecHitsEnergy_zp2nd_ )
       {
	 ESEnergy->cd(2); gPad->SetLogy(); 
	 meESRecHitsEnergy_zp2nd_   ->Draw(); 
	 newmeESRecHitsEnergy_zp2nd_->Draw("same"); 
       }
     if ( meESRecHitsEnergy_zm1st_ && newmeESRecHitsEnergy_zm1st_ )
       {
	 ESEnergy->cd(3); gPad->SetLogy(); 
	 meESRecHitsEnergy_zm1st_   ->Draw(); 
	 newmeESRecHitsEnergy_zm1st_->Draw("same"); 
       }
     if ( meESRecHitsEnergy_zm2nd_ && newmeESRecHitsEnergy_zm2nd_ )
       {
	 ESEnergy->cd(4); gPad->SetLogy(); 
	 meESRecHitsEnergy_zm2nd_   ->Draw(); 
	 newmeESRecHitsEnergy_zm2nd_->Draw("same"); 
       }
     ESEnergy->Print("PreshowerEnergy_compare.eps");
     delete ESEnergy;
   }
 
 
}

